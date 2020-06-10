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

    unsigned int parameterArrayStep = blockDim.x;

    unsigned int threadIndex = threadIdx.x;
    unsigned int sampleIndex = blockIdx.x;
    unsigned int notUsedIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int numberOfParameters = traveltime.numberOfParameters;
    unsigned int numberOfCommonResults = traveltime.numberOfCommonResults;
    unsigned int numberOfTotalParameters = numberOfParameters + numberOfCommonResults;

    unsigned int samplesPerTrace = gatherData.samplesPerTrace;

    if (sampleIndex < samplesPerTrace) {

        gpu_semblance_compute_data_t semblanceCompute;
        gpu_traveltime_parameter_t travelTimeThreadData;

        unsigned int notUsedCount = 0;
        unsigned int usedCount = 0;

        threadSemblanceData[threadIndex] = 0;
        threadSemblanceData[parameterArrayStep + threadIndex] = 0;

        referencePoint.t0 = sampleIndex * gatherData.dtInSeconds;

        travelTimeThreadData.numberOfParameters = numberOfParameters;

        for (unsigned int parameterIndex = 0;
            parameterIndex < numberOfParameters;
            parameterIndex++) {

            unsigned int step, sharedMemoryArrayStep;

            step = parameterIndex * parameterArrayStep;
            sharedMemoryArrayStep = step + traveltime.numberOfCommonResults * parameterArrayStep;

            float parameterValue = parameterArray[step + threadIndex];

            threadSemblanceData[sharedMemoryArrayStep + threadIndex] = parameterValue;

            travelTimeThreadData.semblanceParameters[parameterIndex] = parameterValue;
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

            threadSemblanceData[threadIndex] = sumNumerator / (usedCount * semblanceCompute.denominatorSum);
            threadSemblanceData[parameterArrayStep + threadIndex] = semblanceCompute.linearSum / (usedCount * gatherData.windowSize);
        }

        notUsedCountArray[notUsedIndex] += notUsedCount;

        __syncthreads();

        /* Reduce the best results */
        for (unsigned int s = blockDim.x / 2; s > 0; s = s >> 1) {
            if (threadIndex < s) {
                if (threadSemblanceData[threadIndex] < threadSemblanceData[threadIndex + s]) {
                    for (unsigned int i = 0; i < numberOfTotalParameters; i++) {
                        unsigned step = i * parameterArrayStep;
                        threadSemblanceData[step + threadIndex] = threadSemblanceData[step + threadIndex + s];
                    }
                }
            }
            __syncthreads();
        }

        if (threadIndex == 0) {
            if (threadSemblanceData[0] > resultArray[sampleIndex]) {
                for (unsigned int i = 0; i < numberOfTotalParameters; i++) {
                    unsigned int resultArrayStep = i * samplesPerTrace;
                    unsigned int sharedMemoryArrayStep = i * parameterArrayStep;
                    resultArray[resultArrayStep + sampleIndex] = threadSemblanceData[sharedMemoryArrayStep];
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
