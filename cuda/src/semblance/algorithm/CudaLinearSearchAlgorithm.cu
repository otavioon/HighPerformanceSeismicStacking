#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/algorithm/CudaLinearSearchAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/linear_search.h"

#include <numeric>
#include <sstream>
#include <stdexcept>

using namespace std;

CudaLinearSearchAlgorithm::CudaLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : LinearSearchAlgorithm(traveltime, context, dataBuilder) {
}

void CudaLinearSearchAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

    LOGI("Computing semblance for m0 = " << m0);

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    gpu_gather_data_t kernelData = gather->getGpuGatherData();

    gpu_traveltime_data_t kernelTraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    unsigned int sharedMemSizeCount =
        traveltime->getNumberOfResults() * threadCount * static_cast<unsigned int>(sizeof(float));

    LOGD("threadCount = " << threadCount);
    LOGD("getNumberOfResults = " << traveltime->getNumberOfResults());
    LOGD("sharedMemSizeCount = " << sharedMemSizeCount);

    kernelLinearSearch<<< dimGrid, threadCount, sharedMemSizeCount >>>(
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]),
        filteredTracesCount,
        kernelData,
        kernelReferencePoint,
        kernelTraveltime,
        CUDA_DEV_PTR(deviceParameterArray),
        CUDA_DEV_PTR(deviceResultArray),
        CUDA_DEV_PTR(deviceNotUsedCountArray)
    );

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetLastError());
}

void CudaLinearSearchAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    CUDA_ASSERT(cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char)));

    dim3 dimGrid(traceCount / threadCount + 1);

    LOGI("Using " << dimGrid.x << " blocks for traces filtering (threadCount = "<< threadCount << ")");

    gpu_gather_data_t gatherData = gather->getGpuGatherData();

    gpu_traveltime_data_t kernelTraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    switch (traveltime->getModel()) {
        case CMP:
        case ZOCRS:
            LOGD("Executing filterMidpointDependentTraces<<<>>> kernel");
            filterMidpointDependentTraces<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                traveltime->toGpuData(),
                gather->getApm(),
                m0
            );
            break;

        case OCT:
            LOGD("Executing filterOutTracesForOffsetContinuationTrajectoryAndLinearSearch<<<>>> kernel");
            filterOutTracesForOffsetContinuationTrajectoryAndLinearSearch<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                deviceUsedTraceMaskArray,
                traceCount,
                gatherData,
                kernelReferencePoint,
                kernelTraveltime,
                CUDA_DEV_PTR(deviceParameterArray),
                getParameterArrayStep()
            );
            break;
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    copyOnlySelectedTracesToDevice(usedTraceMask);
}
