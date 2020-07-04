#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>

using namespace std;

LinearSearchAlgorithm::LinearSearchAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : ComputeAlgorithm("linear-search", model, context, dataBuilder),
    discretizationGranularity(model->getNumberOfParameters()),
    discretizationDivisor(model->getNumberOfParameters()),
    discretizationStep(model->getNumberOfParameters()) {
}

void LinearSearchAlgorithm::computeSemblanceAndParametersForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfSamplesPerTrace = gather->getSamplesPerTrace();
    unsigned int parameterArrayStep = getParameterArrayStep();
    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    vector<float> tempParameterArray(numberOfParameters * parameterArrayStep);

    deviceContext->activate();

    deviceResultArray->reset();
    deviceNotUsedCountArray->reset();

    chrono::duration<double> selectionExecutionTime = chrono::duration<double>::zero();
    chrono::duration<double> totalExecutionTime = chrono::duration<double>::zero();

    if (traveltime->getTraveltimeWord() == "oct") {

        vector<float> parameterSampleArray(parameterArrayStep * numberOfParameters);

        default_random_engine generator;

        for (unsigned int prmtr = 0; prmtr < numberOfParameters; prmtr++) {

            float min = traveltime->getLowerBoundForParameter(prmtr);
            float max = traveltime->getUpperBoundForParameter(prmtr);

            uniform_real_distribution<float> uniformDist(min, max);

            unsigned int step = prmtr * parameterArrayStep;

            for (unsigned int idx = 0; idx < parameterArrayStep; idx++) {
                parameterSampleArray[step + idx] = uniformDist(generator);
            }
        }

        deviceParameterArray->copyFrom(parameterSampleArray);
    }

    MEASURE_EXEC_TIME(selectionExecutionTime, selectTracesToBeUsedForMidpoint(m0));

    LOGI("parameterArrayStep = " << parameterArrayStep);
    LOGI("totalNumberOfParameters = " << totalNumberOfParameters);

    for (unsigned int i = 0; i < totalNumberOfParameters; i++) {

        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            unsigned int tempParameterIndex = parameterIndex * parameterArrayStep + i % threadCount;
            tempParameterArray[tempParameterIndex] = getParameterValueAt(i, parameterIndex);
        }

        if ((i + 1) % threadCount == 0 || (i + 1) == totalNumberOfParameters) {

            if ((i + 1) == totalNumberOfParameters) {
                changeThreadCountTemporarilyTo((i + 1) % threadCount);
            }
            else if ((i + 1 + threadCount) >= totalNumberOfParameters) {
                /* Next time we'll be running kernel with a smaller # of threads, so update parameterArrayStep. */
                LOGI("Changing parameterArrayStep to " << totalNumberOfParameters % threadCount);
                parameterArrayStep = totalNumberOfParameters % threadCount;
            }

            deviceParameterArray->copyFrom(tempParameterArray);

            MEASURE_EXEC_TIME(totalExecutionTime, computeSemblanceAtGpuForMidpoint(m0));
        }
    }

    restoreThreadCount();

    deviceResultArray->pasteTo(computedResults);

    unsigned long totalUsedTracesCount;
    totalUsedTracesCount = static_cast<unsigned long>(filteredTracesCount) *
            static_cast<unsigned long>(numberOfSamplesPerTrace) *
            static_cast<unsigned long>(totalNumberOfParameters);

    saveStatisticalResults(totalUsedTracesCount, totalExecutionTime, selectionExecutionTime);
}

float LinearSearchAlgorithm::getParameterValueAt(unsigned int iterationNumber, unsigned int p) const {
    unsigned int step = (iterationNumber / discretizationDivisor[p]) % discretizationGranularity[p];
    return traveltime->getLowerBoundForParameter(p) + static_cast<float>(step) * discretizationStep[p];
}

unsigned int LinearSearchAlgorithm::getTotalNumberOfParameters() const {
    return accumulate(
            discretizationGranularity.begin(),
            discretizationGranularity.end(),
            1, multiplies<unsigned int>()
        );
}

void LinearSearchAlgorithm::setDiscretizationDivisorForParameter(unsigned int p, unsigned int d) {
    if (p >= traveltime->getNumberOfParameters()) {
        throw invalid_argument("Parameter index is out of bounds");
    }
    discretizationDivisor[p] = d;
}

void LinearSearchAlgorithm::setDiscretizationGranularityForParameter(
    unsigned int parameterIndex,
    unsigned int granularity) {

    if (parameterIndex >= traveltime->getNumberOfParameters()) {
        throw invalid_argument("Parameter index is out of bounds");
    }

    discretizationGranularity[parameterIndex] = granularity;
}

void LinearSearchAlgorithm::setUp() {
    setupArrays();

    copyGatherDataToDevice();

    setupDiscretizationSteps();
}

unsigned int LinearSearchAlgorithm::getParameterArrayStep() const {
    return min(getTotalNumberOfParameters(), threadCount);
}

void LinearSearchAlgorithm::setupArrays() {
    ostringstream stringStream;

    Gather* gather = Gather::getInstance();

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfResults = traveltime->getNumberOfResults();
    unsigned int numberOfSamples = gather->getSamplesPerTrace();
    unsigned int parameterArrayStep = getParameterArrayStep();

    LOGI("Allocating data for deviceParameterArray [" << numberOfParameters * parameterArrayStep << " elements]");

    deviceParameterArray.reset(dataFactory->build(numberOfParameters * parameterArrayStep, deviceContext));

    LOGI("Allocating data for deviceNotUsedCountArray [" << numberOfSamples * parameterArrayStep << " elements]");

    deviceNotUsedCountArray.reset(dataFactory->build(numberOfSamples * parameterArrayStep, deviceContext));

    LOGI("Allocating data for deviceResultArray [" << numberOfResults * numberOfSamples << " elements]");

    deviceResultArray.reset(dataFactory->build(numberOfResults * numberOfSamples, deviceContext));

    computedResults.resize(numberOfResults * numberOfSamples);
}

void LinearSearchAlgorithm::setupDiscretizationSteps() {
    unsigned int divisor = 1;
    for (unsigned int i = 0; i < traveltime->getNumberOfParameters(); i++) {

        float lowerBound = traveltime->getLowerBoundForParameter(i);
        float upperBound = traveltime->getUpperBoundForParameter(i);

        discretizationDivisor[i] = divisor;
        divisor *= discretizationGranularity[i];

        discretizationStep[i] = (upperBound - lowerBound) / static_cast<float>(discretizationGranularity[i]);
    }
}

const string LinearSearchAlgorithm::toString() const {
    ostringstream stringStream;

    stringStream << "Total number of parameters = " << getTotalNumberOfParameters() << endl;

    for (unsigned int i = 0; i < traveltime->getNumberOfParameters(); i++) {
        stringStream << "# of discrete values for " << traveltime->getDescriptionForParameter(i);
        stringStream << " = " << discretizationGranularity[i] << endl;
    }

    stringStream << "Thread count = " << threadCount << endl;

    return stringStream.str();
}
