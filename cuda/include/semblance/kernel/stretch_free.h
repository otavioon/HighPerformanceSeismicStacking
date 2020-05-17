#ifndef CUDA_KERNEL_STRETCH_FREE_H
#define CUDA_KERNEL_STRETCH_FREE_H

#include "common/include/gpu/interface.h"

__global__
void kernelStretchFree( const float *samples,
                        const float *midpoint,
                        const float *halfoffset,
                        unsigned int traceCount,
                        const float *inputParameters,
                        gpu_gather_data_t gatherData,
                        gpu_reference_point_t referencePoint,
                        gpu_traveltime_data_t traveltime,
                        /* Parameter arrays */
                        const float *parameterArray,
                        /* Output arrays */
                        float *resultArray,
                        /* Missed traces array */
                        float* notUsedCountArray
);

__global__
void filterOutTracesForOffsetContinuationTrajectoryAndStretchFree(
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