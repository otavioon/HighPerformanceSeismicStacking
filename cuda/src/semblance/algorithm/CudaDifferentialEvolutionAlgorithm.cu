#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/algorithm/CudaDifferentialEvolutionAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/differential_evolution.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>

using namespace std;

CudaDifferentialEvolutionAlgorithm::CudaDifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int gen,
    unsigned int ind
) : DifferentialEvolutionAlgorithm(model, context, dataBuilder, gen, ind) {
}

CudaDifferentialEvolutionAlgorithm::~CudaDifferentialEvolutionAlgorithm() {
    CUDA_ASSERT(cudaFree(st));
}

void CudaDifferentialEvolutionAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    gpu_gather_data_t kernelData = gather->getGpuGatherData();

    gpu_traveltime_data_t kerneltraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    kernelDifferentialEvolution<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]),
        filteredTracesCount,
        kernelData,
        kernelReferencePoint,
        kerneltraveltime,
        CUDA_DEV_PTR(deviceParameterArray),
        CUDA_DEV_PTR(deviceResultArray),
        CUDA_DEV_PTR(deviceNotUsedCountArray)
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    LOGI("Selecting traces for m0 = " << m0);

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    CUDA_ASSERT(cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(char)));
    CUDA_ASSERT(cudaMemset(deviceUsedTraceMaskArray, 0, traceCount * sizeof(unsigned char)))

    dim3 dimGrid(static_cast<int>(ceil(traceCount / threadCount)));

    gpu_gather_data_t gatherData = gather->getGpuGatherData();

    gpu_traveltime_data_t kerneltraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    chrono::duration<double> kernelExecutionTime = chrono::duration<double>::zero();
    chrono::duration<double> copyTime = chrono::duration<double>::zero();

    switch (traveltime->getModel()) {
        case CMP:
        case ZOCRS:
            MEASURE_EXEC_TIME(kernelExecutionTime, (filterMidpointDependentTraces<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                kerneltraveltime,
                gather->getApm(),
                m0
            )));
            break;

        case OCT:
            MEASURE_EXEC_TIME(kernelExecutionTime, (filterOutTracesForOffsetContinuationTrajectoryAndDifferentialEvolution<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                deviceUsedTraceMaskArray,
                traceCount,
                gatherData,
                kernelReferencePoint,
                kerneltraveltime,
                CUDA_DEV_PTR(deviceParameterArray),
                individualsPerPopulation,
                getParameterArrayStep()
            )));
            break;
    }

    LOGI("Execution time for filtering kernel is " << kernelExecutionTime.count() << "s");

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}

void CudaDifferentialEvolutionAlgorithm::setupRandomSeedArray() {

    Gather* gather = Gather::getInstance();

    deviceContext->activate();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();

    dim3 dimGrid(samplesPerTrace);

    CUDA_ASSERT(cudaMalloc(&st, samplesPerTrace * individualsPerPopulation * sizeof(curandState)));

    srand(static_cast<unsigned int>(time(NULL)));

    kernelSetupRandomSeedArray<<< dimGrid, individualsPerPopulation >>>(st, rand());

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::startAllPopulations() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    kernelStartPopulations<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(min),
        CUDA_DEV_PTR(max),
        st,
        numberOfParameters
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    fx->reset();
    fu->reset();
}

void CudaDifferentialEvolutionAlgorithm::mutateAllPopulations() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    kernelmutateAllPopulations<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(v),
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(min),
        CUDA_DEV_PTR(max),
        st,
        numberOfParameters
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::crossoverPopulationIndividuals() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    kernelCrossoverPopulationIndividuals<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(u),
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(v),
        st,
        numberOfParameters
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::advanceGeneration() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    kernelAdvanceGeneration<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(fx),
        CUDA_DEV_PTR(u),
        CUDA_DEV_PTR(fu),
        numberOfParameters,
        numberOfCommonResults
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::selectBestIndividuals(vector<float>& resultArrays) {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfResults = traveltime->getNumberOfResults();
    unsigned int sharedMemCount = numberOfResults * individualsPerPopulation * static_cast<unsigned int>(sizeof(float));

    deviceResultArray.reset(dataFactory->build(numberOfResults * gather->getSamplesPerTrace(), deviceContext));

    kernelSelectBestIndividuals<<< dimGrid, individualsPerPopulation, sharedMemCount >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(fx),
        CUDA_DEV_PTR(deviceResultArray),
        traveltime->getNumberOfParameters(),
        traveltime->getNumberOfCommonResults(),
        gather->getSamplesPerTrace()
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    deviceResultArray->pasteTo(resultArrays);
}
