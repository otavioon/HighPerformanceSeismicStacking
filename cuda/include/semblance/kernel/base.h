#ifndef CUDA_KERNEL_BASE_H
#define CUDA_KERNEL_BASE_H

#include "common/include/gpu/interface.h"
#include <cuda.h>

__global__ void filterMidpointDependentTraces(
                float* midpointArray,
                unsigned int traceCount,
                unsigned char* usedTraceMaskArray,
                gpu_traveltime_data_t model,
                float apm,
                float m0);

__device__ int shouldUseTrace(
                float m,
                float h,
                gpu_gather_data_t kernelData,
                gpu_reference_point_t kernelReferencePoint,
                gpu_traveltime_data_t model,
                gpu_traveltime_parameter_t parameters,
                void* out);

__device__ enum gpu_error_code computeDisplacedMidpoint(
                float h,
                gpu_reference_point_t referencePoint,
                gpu_traveltime_parameter_t parameters,
                float* mh);

__device__ enum gpu_error_code computeTime(
                float m,
                float h,
                gpu_reference_point_t referencePoint,
                gpu_traveltime_data_t model,
                gpu_traveltime_parameter_t parameters,
                float* out);

__device__ enum gpu_error_code computeSemblance(
                const float *samples,
                float t,
                gpu_gather_data_t kernelData,
                gpu_semblance_compute_data_t *computeData);
#endif