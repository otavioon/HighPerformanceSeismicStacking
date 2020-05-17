#include "cuda/include/semblance/kernel/base.h"

__global__
void filterMidpointDependentTraces(
    float* midpointArray,
    unsigned int traceCount,
    unsigned char* usedTraceMaskArray,
    gpu_traveltime_data_t model,
    float apm, float m0
) {
    unsigned int traceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (traceIndex < traceCount) {
        switch (model.traveltime) {
            case CMP:
                usedTraceMaskArray[traceIndex] = (m0 == midpointArray[traceIndex]);
                break;
            case ZOCRS:
                usedTraceMaskArray[traceIndex] = fabs(m0 - midpointArray[traceIndex]) <= apm;
                break;
            default:
                usedTraceMaskArray[traceIndex] = 0;
        }
    }
}

__device__
int shouldUseTrace(
    float m, float h,
    gpu_gather_data_t kernelData,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t model,
    gpu_traveltime_parameter_t parameters,
    void* out
) {
    float mh;

    enum gpu_error_code errorCode;

    switch (model.traveltime) {
        case CMP:
        case ZOCRS:
            return 1;
        case OCT:
            errorCode = computeDisplacedMidpoint(
                h,
                referencePoint,
                parameters,
                &mh
            );

            if (errorCode == NO_ERROR) {
                if (out && fabs(m - mh) <= kernelData.apm) {
                    *((float *) out) = mh;
                    return 1;
                }
            }
            break;
    }

    return 0;
}

__device__
enum gpu_error_code computeDisplacedMidpoint(
    float h,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_parameter_t parameters,
    float* mh
) {
    float velocity = parameters.semblanceParameters[0];
    float slope = parameters.semblanceParameters[1];

    float h0 = referencePoint.h0;
    float t0 = referencePoint.t0;

    if (!velocity) {
        return DIVISION_BY_ZERO;
    }

    float   theta = t0 * slope,
            gamma = 2 * sqrt(h * h + h0 * h0),
            tn0_sq = t0 * t0 - 4 * h0 * h0 / (velocity * velocity);

    float   theta_sq = theta * theta,
            gamma_sq = gamma * gamma,
            tn0_quad = tn0_sq * tn0_sq,
            sqrt_arg = tn0_quad * tn0_quad + tn0_quad * theta_sq * gamma_sq  +
                                16 * h * h * h0 * h0 * theta_sq * theta_sq;

    if (sqrt_arg < 0) {
        return NEGATIVE_SQUARED_ROOT;
    }

    sqrt_arg = theta_sq * gamma_sq + 2 * tn0_quad + 2 * sqrt(sqrt_arg);

    if (sqrt_arg == 0) {
        return DIVISION_BY_ZERO;
    }

    *mh = referencePoint.m0 + 2 * theta * (h * h - h0 * h0) / sqrt(sqrt_arg);

    return NO_ERROR;
}

__device__
enum gpu_error_code computeTime(
    float m, float h,
    gpu_reference_point_t referencePoint,
    gpu_traveltime_data_t model,
    gpu_traveltime_parameter_t parameters,
    float* out
) {
    float t0 = referencePoint.t0;

    switch (model.traveltime) {
        case CMP:
            if (parameters.semblanceParameters[0] != 0) {
                float v = parameters.semblanceParameters[0];
                *out = sqrt(t0 * t0 + 4 * h * h / (v * v));
                return NO_ERROR;
            }

            return DIVISION_BY_ZERO;

        case ZOCRS:
            if (parameters.semblanceParameters[0] != 0) {

                float v = parameters.semblanceParameters[0];
                float a = parameters.semblanceParameters[1];
                float b = parameters.semblanceParameters[2];

                float tmp, dm;

                dm = m - referencePoint.m0;
                tmp = t0 + a * dm,
                tmp = tmp * tmp +  b * dm * dm + 4 * h * h / (v * v);

                if (tmp >= 0) {
                    *out = sqrt(tmp);
                    return NO_ERROR;
                }

                return NEGATIVE_SQUARED_ROOT;
            }

            return DIVISION_BY_ZERO;

        case OCT:
            if (parameters.semblanceParameters[0] != 0) {

                float tn0_sq, tn_sq, w_sqrt_1, w_sqrt_2, u, sqrt_arg, ah, th;

                float v = parameters.semblanceParameters[0];
                float a = parameters.semblanceParameters[1];

                float m0 = referencePoint.m0;
                float h0 = referencePoint.h0;
                float t0 = referencePoint.t0;
                float mh = parameters.mh;

                tn0_sq = t0 * t0 - 4 * h0 * h0 / (v * v);

                w_sqrt_1 = (h + h0) * (h + h0) - (mh - m0) * (mh - m0);
                w_sqrt_2 = (h - h0) * (h - h0) - (mh - m0) * (mh - m0);

                if (w_sqrt_1 < 0 || w_sqrt_2 < 0) {
                    return NEGATIVE_SQUARED_ROOT;
                }

                u = sqrt(w_sqrt_1) + sqrt(w_sqrt_2);

                th = t0;
                if (fabs(h) > fabs(h0)) {

                    if (!u) {
                        return DIVISION_BY_ZERO;
                    }

                    sqrt_arg = 4 * h * h / (v * v) + 4 * h * h / (u * u) * tn0_sq;

                    if (sqrt_arg < 0) {
                        return NEGATIVE_SQUARED_ROOT;
                    }

                    th = sqrt(sqrt_arg);
                }
                else if (fabs(h) < fabs(h0)) {

                    if (!h0) {
                        return DIVISION_BY_ZERO;
                    }

                    sqrt_arg = 4 * h * h / (v * v) + u * u / (4 * h0 * h0) * tn0_sq;

                    if (sqrt_arg < 0) {
                        return NEGATIVE_SQUARED_ROOT;
                    }

                    th = sqrt(sqrt_arg);
                }

                tn_sq = th * th - 4 * h * h / (v * v);

                if (!th || !tn0_sq) {
                    return DIVISION_BY_ZERO;
                }

                ah = (t0 * tn_sq) / (th * tn0_sq) * a;

                *out = th + ah * (m - mh);

                return NO_ERROR;
            }

            return DIVISION_BY_ZERO;
    }

    return INVALID_MODEL;
}

__device__
enum gpu_error_code computeSemblance(
    const float *samples,
    float t,
    gpu_gather_data_t kernelData,
    gpu_semblance_compute_data_t *computeData
) {
    float tIndex, dt;
    int kIndex;

    tIndex = t / kernelData.dtInSeconds;
    kIndex = static_cast<int>(tIndex);

    dt = tIndex - static_cast<float>(kIndex);

    if ((kIndex - kernelData.tauIndexDisplacement >= 0) &&
        (kIndex + kernelData.tauIndexDisplacement + 1 < kernelData.samplesPerTrace)) {

        for (int j = 0; j < kernelData.windowSize; j++) {

            int k = kIndex - kernelData.tauIndexDisplacement + j;

            float y1 = samples[k + 1];
            float y0 = samples[k];
            float u = (y1 - y0) * dt + y0;

            computeData->numeratorComponents[j] += u;
            computeData->linearSum += u;
            computeData->denominatorSum += u * u;
        }

        return NO_ERROR;
    }

    return INVALID_RANGE;
}
