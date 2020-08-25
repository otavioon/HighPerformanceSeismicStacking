#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/differential_evolution.h"

__global__
void kernelDifferentialEvolution(
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
) {
    unsigned int parameterIndex = threadIdx.x;
    unsigned int sampleIndex = blockIdx.x;
    unsigned int notUsedIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int individualPerPopulation = blockDim.x;

    if (sampleIndex < gatherData.samplesPerTrace) {

        unsigned int arrayStep = individualPerPopulation * traveltime.numberOfParameters;

        gpu_semblance_compute_data_t semblanceCompute;
        gpu_traveltime_parameter_t travelTimeThreadData;

        unsigned int notUsedCount = 0;
        unsigned int usedCount = 0;

        referencePoint.t0 = sampleIndex * gatherData.dtInSeconds;

        travelTimeThreadData.numberOfParameters = traveltime.numberOfParameters;

        for (unsigned int i = 0; i < traveltime.numberOfParameters; i++) {
            unsigned int step = sampleIndex * arrayStep + i * individualPerPopulation;
            travelTimeThreadData.semblanceParameters[i] = parameterArray[step + parameterIndex];
        }

        semblanceCompute.denominatorSum = semblanceCompute.linearSum = 0;
        memset(semblanceCompute.numeratorComponents, 0, MAX_WINDOW_SIZE * sizeof(float));

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {

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

            float t;
            if (computeTime(m, h, referencePoint, traveltime, travelTimeThreadData, &t) == NO_ERROR) {
                if (computeSemblance(traceSamples, t, gatherData, &semblanceCompute) == NO_ERROR) {
                    usedCount++;
                }
            }
        }

        if (usedCount) {

            unsigned int resultArrayStep = individualPerPopulation * traveltime.numberOfCommonResults;
            unsigned int step = sampleIndex * resultArrayStep;

            float sumNumerator = 0;
            for (int j = 0; j < gatherData.windowSize; j++) {
                sumNumerator +=
                    semblanceCompute.numeratorComponents[j] * semblanceCompute.numeratorComponents[j];
            }

            resultArray[step + parameterIndex] =
                    sumNumerator / (usedCount * semblanceCompute.denominatorSum);

            resultArray[step + individualPerPopulation + parameterIndex] =
                semblanceCompute.linearSum / (usedCount * gatherData.windowSize);
        }

        notUsedCountArray[notUsedIndex] += notUsedCount;
    }
}

__global__
void filterOutTracesForOffsetContinuationTrajectoryAndDifferentialEvolution(
    const float *midpointArray,
    const float *halfoffsetArray,
    unsigned char* mustUseTraceArray,
    unsigned int traceCount,
    gpu_gather_data_t gatherData,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t traveltime,
    const float *parameterArray,
    unsigned int parameterCount,
    unsigned int arrayStep
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
                    unsigned int step = sampleIndex * arrayStep + i * parameterCount;
                    travelTimeThreadData.semblanceParameters[i] = parameterArray[step + parameterIndex];
                }

                mustUseTraceArray[traceIndex] = shouldUseTrace(
                    m, h,
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

__global__
void kernelSetupRandomSeedArray(
    curandState *state,
    unsigned int seed
) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__
void kernelStartPopulations(
    float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int numberOfParameters
) {
    unsigned int individualsPerPopulation = blockDim.x;
    unsigned int arrayStep = numberOfParameters * individualsPerPopulation;
    unsigned int t0 = blockIdx.x;
    unsigned int individualIndex = threadIdx.x;
    unsigned int seedIndex = t0 * individualsPerPopulation + individualIndex;

    for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
        float ratio = curand_uniform(&st[seedIndex]);
        unsigned int individualArrayIndex = t0 * arrayStep + parameterIndex * individualsPerPopulation + individualIndex;
        x[individualArrayIndex] = min[parameterIndex] + ratio * (max[parameterIndex] - min[parameterIndex]);
    }
}

__global__
void kernelmutateAllPopulations(
    float* v,
    const float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int numberOfParameters
) {
    unsigned int individualsPerPopulation = blockDim.x;
    unsigned int arrayStep = numberOfParameters * individualsPerPopulation;
    unsigned int t0 = blockIdx.x;
    unsigned int individualIndex = threadIdx.x;
    unsigned int seedIndex = t0 * individualsPerPopulation + individualIndex;

    for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {

        float p1 = curand_uniform(&st[seedIndex]);
        float p2 = curand_uniform(&st[seedIndex]);
        float p3 = curand_uniform(&st[seedIndex]);

        unsigned int popIndex = t0 * arrayStep + parameterIndex * individualsPerPopulation;

        unsigned int r1 = popIndex + static_cast<unsigned int>(p1 * static_cast<float>(individualsPerPopulation - 1));
        unsigned int r2 = popIndex + static_cast<unsigned int>(p2 * static_cast<float>(individualsPerPopulation - 1));
        unsigned int r3 = popIndex + static_cast<unsigned int>(p3 * static_cast<float>(individualsPerPopulation - 1));

        float newIndividual = x[r1] + F_FAC * (x[r2] - x[r3]);
        newIndividual = fminf(newIndividual, max[parameterIndex]);
        newIndividual = fmaxf(newIndividual, min[parameterIndex]);

        v[popIndex + individualIndex] = newIndividual;
    }
}

__global__
void kernelCrossoverPopulationIndividuals(
    float* u,
    const float* x,
    const float* v,
    curandState *st,
    unsigned int numberOfParameters
) {
    unsigned int individualsPerPopulation = blockDim.x;
    unsigned int arrayStep = numberOfParameters * individualsPerPopulation;
    unsigned int t0 = blockIdx.x;
    unsigned int individualIndex = threadIdx.x;
    unsigned int seedIndex = t0 * individualsPerPopulation + individualIndex;

    unsigned int l = static_cast<unsigned int>(
        curand_uniform(&st[seedIndex]) * static_cast<float>(numberOfParameters - 1)
    );

    for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {

        float r = curand_uniform(&st[seedIndex]);

        unsigned int individualArrayIndex = t0 * arrayStep + parameterIndex * individualsPerPopulation + individualIndex;

        if (r > CR && l != parameterIndex) {
            u[individualArrayIndex] = x[individualArrayIndex];
        }
        else {
            u[individualArrayIndex] = v[individualArrayIndex];
        }
    }
}

__global__
void kernelAdvanceGeneration(
    float* x,
    float* fx,
    const float* u,
    const float* fu,
    unsigned int numberOfParameters,
    unsigned int numberOfCommonResults
) {
    unsigned int individualsPerPopulation = blockDim.x;
    unsigned int parameterArrayStep = numberOfParameters * individualsPerPopulation;
    unsigned int resultArrayStep = numberOfCommonResults * individualsPerPopulation;

    unsigned int t0 = blockIdx.x;
    unsigned int individualIndex = threadIdx.x;

    unsigned int resultSemblIndex = t0 * resultArrayStep + individualIndex;

    if (fu[resultSemblIndex] > fx[resultSemblIndex]) {

        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {

            unsigned int individualArrayIndex =
                t0 * parameterArrayStep + parameterIndex * individualsPerPopulation + individualIndex;

            x[individualArrayIndex] = u[individualArrayIndex];
        }

        for (unsigned int resultIndex = 0; resultIndex < numberOfCommonResults; resultIndex++) {

            unsigned int individualArrayIndex =
                t0 * resultArrayStep + resultIndex * individualsPerPopulation + individualIndex;

            fx[individualArrayIndex] = fu[individualArrayIndex];
        }
    }
}

__global__
void kernelSelectBestIndividuals(
    const float* x,
    const float* fx,
    float* resultArray,
    unsigned int numberOfParameters,
    unsigned int numberOfCommonResults,
    unsigned int numberOfSamples
) {
    extern __shared__ float sharedMemArray[];

    unsigned int individualsPerPopulation = blockDim.x;
    unsigned int parameterArrayStep = numberOfParameters * individualsPerPopulation;
    unsigned int resultArrayStep = numberOfCommonResults * individualsPerPopulation;
    unsigned int numberOfTotalParameters = numberOfParameters + numberOfCommonResults;

    unsigned int t0 = blockIdx.x;
    unsigned int individualIndex = threadIdx.x;

    for (unsigned int resultIndex = 0;
        resultIndex < numberOfCommonResults;
        resultIndex++
    ) {
        unsigned int individualArrayIndex =
                t0 * resultArrayStep +
                resultIndex * individualsPerPopulation +
                individualIndex;

        unsigned int sharedMemIndex =
            resultIndex * individualsPerPopulation +
            individualIndex;

        sharedMemArray[sharedMemIndex] = fx[individualArrayIndex];
    }

    for (unsigned int parameterIndex = 0;
        parameterIndex < numberOfParameters;
        parameterIndex++
    ) {

        unsigned int individualArrayIndex =
            t0 * parameterArrayStep +
            parameterIndex * individualsPerPopulation +
            individualIndex;

        unsigned int sharedMemIndex =
            (parameterIndex + numberOfCommonResults) * individualsPerPopulation +
            individualIndex;

        sharedMemArray[sharedMemIndex] = x[individualArrayIndex];
    }

    __syncthreads();
    /* TODO: Select next power of 2 greater than blockDim.x */
    /* Reduce the best results */
    for (unsigned int sharedMemOffset = blockDim.x / 2;
        sharedMemOffset > 0;
        sharedMemOffset = sharedMemOffset >> 1
    ) {
        if (individualIndex < sharedMemOffset) {

            if (sharedMemArray[individualIndex] < sharedMemArray[individualIndex + sharedMemOffset]) {

                for (unsigned int parameterIndex = 0;
                    parameterIndex < numberOfTotalParameters;
                    parameterIndex++
                ) {
                    unsigned int sharedMemIndex =
                        parameterIndex * individualsPerPopulation +
                        individualIndex;

                    sharedMemArray[sharedMemIndex] =
                        sharedMemArray[sharedMemIndex + sharedMemOffset];
                }
            }
        }
        __syncthreads();
    }

    if (individualIndex == 0) {
        for (unsigned int parameterIndex = 0;
            parameterIndex < numberOfTotalParameters;
            parameterIndex++
        ) {
            unsigned int sharedMemIndex = parameterIndex * individualsPerPopulation;

            unsigned int resultArrayIndex = parameterIndex * numberOfSamples;

            resultArray[resultArrayIndex + t0] = sharedMemArray[sharedMemIndex];
        }
    }
}