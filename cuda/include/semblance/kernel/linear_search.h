#ifndef CUDA_KERNEL_LINEAR_SEARCH_H
#define CUDA_KERNEL_LINEAR_SEARCH_H

#include "common/include/gpu/interface.h"
#include <cuda.h>

__global__
void kernelLinearSearch(
    const float *samples,
    const float *midpoint,
    const float *halfoffset,
    unsigned int traceCount,
    gpu_gather_data_t gatherData,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t traveltime,
    const float *parameterArray,
    float *resultArray,
    float* notUsedCountArray
);

__global__
void filterOutTracesForOffsetContinuationTrajectoryAndLinearSearch(
    const float *midpointArray,
    const float *halfoffsetArray,
    unsigned char* mustUseTraceArray,
    unsigned int traceCount,
    gpu_gather_data_t gatherData,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t traveltime,
    const float *parameterArray,
    unsigned int parameterCount
);
#endif