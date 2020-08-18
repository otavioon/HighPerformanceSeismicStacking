#include "common/include/execution/Utils.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/algorithm/CudaStretchFreeAlgorithm.hpp"
#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/stretch_free.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace std;

CudaStretchFreeAlgorithm::CudaStretchFreeAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    const vector<string>& files
) : StretchFreeAlgorithm(traveltime, context, dataBuilder, files) {
}

void CudaStretchFreeAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    gpu_gather_data_t kernelData = gather->getGpuGatherData();

    gpu_traveltime_data_t kerneltraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    unsigned int sharedMemSizeCount =
        (traveltime->getNumberOfCommonResults() + 1) * threadCount * static_cast<unsigned int>(sizeof(float));

    kernelStretchFree<<< dimGrid, threadCount, sharedMemSizeCount >>>(
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]),
        filteredTracesCount,
        CUDA_DEV_PTR(nonStretchFreeParameters[m0]),
        kernelData,
        kernelReferencePoint,
        kerneltraveltime,
        CUDA_DEV_PTR(deviceParameterArray),
        CUDA_DEV_PTR(deviceResultArray),
        CUDA_DEV_PTR(deviceNotUsedCountArray)
    );

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetLastError());
}

void CudaStretchFreeAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(char));

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
            MEASURE_EXEC_TIME(kernelExecutionTime, filterMidpointDependentTraces<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                traveltime->toGpuData(),
                gather->getApm(),
                m0
            ));
            break;

        case OCT:
            MEASURE_EXEC_TIME(kernelExecutionTime, filterOutTracesForOffsetContinuationTrajectoryAndDifferentialEvolution<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                deviceUsedTraceMaskArray,
                traceCount,
                gatherData,
                kernelReferencePoint,
                kerneltraveltime,
                CUDA_DEV_PTR(nonStretchFreeParameters[m0]),
                gather->getSamplesPerTrace()
            ));
            break;
    }

    LOGI("Execution time for filtering kernel is " << kernelExecutionTime.count() << "s");

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}
