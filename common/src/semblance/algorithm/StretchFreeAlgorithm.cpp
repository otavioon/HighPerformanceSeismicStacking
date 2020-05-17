#include "common/include/model/Trace.hpp"
#include "common/include/semblance/algorithm/StretchFreeAlgorithm.hpp"

#include <fstream>
#include <sstream>

#define NMAX 10

using namespace std;

StretchFreeAlgorithm::StretchFreeAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    const vector<string>& files
) : ComputeAlgorithm("strecth-free", model, context, dataBuilder),
    parameterFileArray(files) {
}

void StretchFreeAlgorithm::computeSemblanceAndParametersForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();
    unsigned int numberOfSamplesPerTrace = gather->getSamplesPerTrace();
    unsigned int parameterArrayStep = getParameterArrayStep();

    vector<float> tempParameterArray(parameterArrayStep);

    deviceResultArray->reset();
    deviceNotUsedCountArray->reset();

    selectTracesToBeUsedForMidpoint(m0);

    chrono::steady_clock::time_point kernelStartTimestamp;
    chrono::duration<double> totalExecutionTime = chrono::duration<double>::zero();

    unsigned int idx = 0;
    for (int n = -NMAX; n <= NMAX; n++, idx++) {

        tempParameterArray[idx % parameterArrayStep] = static_cast<float>(n);

        if ((idx + 1) % threadCount == 0 || (idx + 1) == totalNumberOfParameters) {

            if ((idx + 1) == totalNumberOfParameters) {
                changeThreadCountTemporarilyTo((idx + 1) % threadCount);
            }

            deviceParameterArray->copyFrom(tempParameterArray);

            kernelStartTimestamp = chrono::steady_clock::now();
            computeSemblanceAtGpuForMidpoint(m0);
            totalExecutionTime += chrono::steady_clock::now() - kernelStartTimestamp;
        }
    }

    restoreThreadCount();

    deviceResultArray->pasteTo(computedResults);

    unsigned long totalUsedTracesCount;
    totalUsedTracesCount = static_cast<unsigned long>(filteredTracesCount) *
            static_cast<unsigned long>(numberOfSamplesPerTrace) *
            static_cast<unsigned long>(totalNumberOfParameters);

    saveStatisticalResults(totalUsedTracesCount, totalExecutionTime);
}

unsigned int StretchFreeAlgorithm::getTotalNumberOfParameters() const {
    /* TODO: improve this */
    return 2 * NMAX + 1;
}

unsigned int StretchFreeAlgorithm::getParameterArrayStep() const {
    return min(getTotalNumberOfParameters(), threadCount);
}

void StretchFreeAlgorithm::setUp() {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfResults = traveltime->getNumberOfCommonResults() + 1;
    unsigned int numberOfSamples = gather->getSamplesPerTrace();
    unsigned int parameterArrayStep = getParameterArrayStep();

    copyGatherDataToDevice();

    deviceParameterArray.reset(dataFactory->build(parameterArrayStep, deviceContext));
    deviceNotUsedCountArray.reset(dataFactory->build(numberOfSamples * parameterArrayStep, deviceContext));
    deviceResultArray.reset(dataFactory->build(numberOfResults * numberOfSamples, deviceContext));

    computedResults.resize(numberOfResults * numberOfSamples);

    readNonStretchedFreeParameterFromFile();
}

const string StretchFreeAlgorithm::toString() const {

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    ostringstream stringStream;

    stringStream << "Total number of parameters = " << getParameterArrayStep() << endl;
    stringStream << "N in [ " << -NMAX << "," << NMAX << "]" << endl;

    stringStream << "Parameter files: " << endl;
    for (unsigned int i = 0; i < numberOfParameters; i++) {
         stringStream << parameterFileArray[i] << endl;
    }

    return stringStream.str();
}

void StretchFreeAlgorithm::readNonStretchedFreeParameterFromFile() {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfSamples = gather->getSamplesPerTrace();

    vector<ifstream> fileArray(numberOfParameters);

    for(unsigned int i = 0; i < numberOfParameters; i++) {
        fileArray[i].open(parameterFileArray[i], ios::binary);
    }

    for (auto it = gather->getCdps().begin(); it != gather->getCdps().end(); it++) {

        float m0 = it->first;

        nonStretchFreeParameters[m0].reset(dataFactory->build(numberOfParameters * numberOfSamples, deviceContext));

        for(unsigned int i = 0; i < numberOfParameters; i++) {
            Trace t;
            t.read(fileArray[i], gather->getAzimuthInRad());
            nonStretchFreeParameters[m0]->copyFromWithOffset(t.getSamples(),  i * numberOfSamples);
        }
    }
}
