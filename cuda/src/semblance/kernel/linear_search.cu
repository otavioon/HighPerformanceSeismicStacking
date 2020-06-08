#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/linear_search.h"

__global__
void kernelLinearSearch(const float *samples,
                        const float *midpoint,
                        const float *halfoffset,
                        unsigned int traceCount,
                        gpu_gather_data_t gatherData,
                        gpu_reference_point_t referencePoint,
                        gpu_traveltime_data_t traveltime,
                        /* Parameter arrays */
                        const float *parameterArray,
                        /* Output arrays */
                        float *resultArray,
                        /* Missed traces array */
                        float* notUsedCountArray
) {
    extern __shared__ float threadSemblanceData[];

    unsigned int arrayStep = blockDim.x;

    unsigned int parameterThreadIndex = threadIdx.x;
    unsigned int sampleIndex = blockIdx.x;
    unsigned int notUsedIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int numberOfParameters = traveltime.numberOfParameters;
    unsigned int numberOfCommonResults = traveltime.numberOfCommonResults;
    unsigned int numberOfTotalParameters =
        numberOfParameters + numberOfCommonResults;

    unsigned int samplesPerTrace = gatherData.samplesPerTrace;

    if (sampleIndex < samplesPerTrace) {

        gpu_semblance_compute_data_t semblanceCompute;
        gpu_traveltime_parameter_t travelTimeThreadData;

        unsigned int notUsedCount = 0;
        unsigned int usedCount = 0;

        threadSemblanceData[parameterThreadIndex] = 0;
        threadSemblanceData[arrayStep + parameterThreadIndex] = 0;

        referencePoint.t0 = sampleIndex * gatherData.dtInSeconds;

        travelTimeThreadData.numberOfParameters = numberOfParameters;

        for (unsigned int parameterIndex = 0;
            parameterIndex < numberOfParameters;
            parameterIndex++) {

            unsigned int step, sharedMemStep;

            step = parameterIndex * arrayStep;
            sharedMemStep = step + traveltime.numberOfCommonResults * arrayStep;

            threadSemblanceData[sharedMemStep + parameterThreadIndex] =
                parameterArray[step + parameterThreadIndex];

            travelTimeThreadData.semblanceParameters[parameterIndex] =
                parameterArray[step + parameterThreadIndex];
        }

        semblanceCompute.denominatorSum = semblanceCompute.linearSum = 0;
        memset(semblanceCompute.numeratorComponents, 0, MAX_WINDOW_SIZE * sizeof(float));

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float t;
            float h = halfoffset[traceIndex];
            float m = midpoint[traceIndex];
            const float *traceSamples = samples + traceIndex * gatherData.samplesPerTrace;

            if (traveltime.traveltime == OCT) {
                if (!shouldUseTrace(
                        m, h,
                        gatherData,
                        referencePoint,
                        traveltime,
                        travelTimeThreadData,
                        &travelTimeThreadData.mh)
                    ) {
                    notUsedCount++;
                    continue;
                }
            }

            if (computeTime(m, h, referencePoint, traveltime, travelTimeThreadData, &t) == NO_ERROR) {
                if (computeSemblance(traceSamples, t, gatherData, &semblanceCompute) == NO_ERROR) {
                    usedCount++;
                }
            }
        }

        if (usedCount) {
            float sumNumerator = 0;
            for (int j = 0; j < gatherData.windowSize; j++) {
                sumNumerator += semblanceCompute.numeratorComponents[j] * semblanceCompute.numeratorComponents[j];
            }

            threadSemblanceData[parameterThreadIndex] =
                sumNumerator / (usedCount * semblanceCompute.denominatorSum);
            threadSemblanceData[arrayStep + parameterThreadIndex] =
                semblanceCompute.linearSum / (usedCount * gatherData.windowSize);
        }

        notUsedCountArray[notUsedIndex] += notUsedCount;

        __syncthreads();

        /* Reduce the best results */
        for (unsigned int s = blockDim.x / 2; s > 0; s = s >> 1) {
            if (parameterThreadIndex < s) {
                if (threadSemblanceData[parameterThreadIndex] < threadSemblanceData[parameterThreadIndex + s]) {
                    for (unsigned int i = 0; i < numberOfTotalParameters; i++) {
                        unsigned step = i * arrayStep;
                        threadSemblanceData[step + parameterThreadIndex] =
                            threadSemblanceData[step + parameterThreadIndex + s];
                    }
                }
            }
            __syncthreads();
        }

        if (parameterThreadIndex == 0) {
            if (threadSemblanceData[0] > resultArray[sampleIndex]) {
                for (unsigned int i = 0; i < numberOfTotalParameters; i++) {
                    unsigned int step = i * samplesPerTrace;
                    unsigned int sharedMemStep = i * arrayStep;
                    resultArray[step + sampleIndex] = threadSemblanceData[sharedMemStep];
                }
            }
        }
    }
}

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
) {
    unsigned int traceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (traceIndex < traceCount) {

        gpu_traveltime_parameter_t travelTimeThreadData;

        float m, h;
        m = midpointArray[traceIndex];
        h = halfoffsetArray[traceIndex];

        mustUseTraceArray[traceIndex] = 0;

        for (unsigned int sampleIndex = 0; !mustUseTraceArray[traceIndex] && sampleIndex < gatherData.samplesPerTrace; sampleIndex++) {

            referencePoint.t0 = sampleIndex * gatherData.dtInSeconds;

            for (unsigned int parameterIndex = 0; !mustUseTraceArray[traceIndex] && parameterIndex < parameterCount; parameterIndex++) {

                for (unsigned int i = 0; i < traveltime.numberOfParameters; i++) {
                    unsigned int step = i * parameterCount;
                    travelTimeThreadData.semblanceParameters[i] = parameterArray[step + parameterIndex];
                }

                mustUseTraceArray[traceIndex] =
                    shouldUseTrace(
                        m,
                        h,
                        gatherData,
                        referencePoint,
                        traveltime,
                        travelTimeThreadData,
                        &travelTimeThreadData.mh
                    );
            }
        }
    }
}
