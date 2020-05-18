#include "common/include/output/Logger.hpp"
#include "cuda/include/semblance/algorithm/CudaDifferentialEvolutionAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"

#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/differential_evolution.h"

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

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernelDifferentialEvolution<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();
}

void CudaDifferentialEvolutionAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(char));

    dim3 dimGrid(traceCount / threadCount + 1);

    gpu_gather_data_t gatherData = gather->getGpuGatherData();

    gpu_traveltime_data_t kerneltraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    switch (traveltime->getModel()) {
        case CMP:
        case ZOCRS:
            filterMidpointDependentTraces<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                kerneltraveltime,
                gather->getApm(),
                m0
            );
            break;

        case OCT:
            filterOutTracesForOffsetContinuationTrajectoryAndDifferentialEvolution<<<dimGrid, threadCount>>>(
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
            );
            break;
    }

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernel launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();

    cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(deviceUsedTraceMaskArray);

    copyOnlySelectedTracesToDevice(usedTraceMask);
}

void CudaDifferentialEvolutionAlgorithm::setupRandomSeedArray() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    srand(static_cast<unsigned int>(time(NULL)));

    kernelSetupRandomSeedArray<<< dimGrid, individualsPerPopulation >>>(
        st,
        rand()
    );

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernelSetupRandomSeedArray<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();
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

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernelStartPopulations<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();

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

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA mutateAllPopulations<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();
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

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernelCrossoverPopulationIndividuals<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();
}

void CudaDifferentialEvolutionAlgorithm::advanceGeneration() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfResults = traveltime->getNumberOfResults();

    kernelAdvanceGeneration<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(fx),
        CUDA_DEV_PTR(u),
        CUDA_DEV_PTR(fu),
        numberOfParameters,
        numberOfResults
    );

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernelAdvanceGeneration<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();
}

void CudaDifferentialEvolutionAlgorithm::selectBestIndividuals(vector<float>& resultArrays) {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfResults = traveltime->getNumberOfResults();
    unsigned int sharedMemCount = numberOfResults * individualsPerPopulation * static_cast<unsigned int>(sizeof(float));

    deviceResultArray.reset(dataFactory->build(numberOfResults * individualsPerPopulation, deviceContext));

    kernelSelectBestIndividuals<<< dimGrid, individualsPerPopulation, sharedMemCount >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(fx),
        CUDA_DEV_PTR(deviceResultArray),
        traveltime->getNumberOfParameters(),
        traveltime->getNumberOfCommonResults(),
        gather->getSamplesPerTrace()
    );

    cudaError_t errorCode = cudaGetLastError();

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA kernelSelectBestIndividuals<<<>>> launch failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    cudaDeviceSynchronize();

    deviceResultArray->pasteTo(resultArrays);
}
