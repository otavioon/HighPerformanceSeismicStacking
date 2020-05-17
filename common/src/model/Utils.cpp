#include "common/include/model/Trace.hpp"
#include "common/include/model/Utils.hpp"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float Utils::halfoffset_axis(trace_info_t traceInfo, Axis axis) {
    float scalco = static_cast<float>(traceInfo.scalco);
    float fscalco = traceInfo.scalco >= 0 ? scalco: -1.0f / scalco;

    switch(axis) {
        case Axis::X:
            return fscalco * static_cast<float>(traceInfo.gx - traceInfo.sx) * 0.5f;
            break;
        case Axis::Y:
            return fscalco * static_cast<float>(traceInfo.gy - traceInfo.sy) * 0.5f;
            break;
    }

    return 0;
}

float Utils::midpoint_axis(trace_info_t traceInfo, Axis axis) {
    float scalco = static_cast<float>(traceInfo.scalco);
    float fscalco = traceInfo.scalco >= 0 ? scalco: -1.0f / scalco;

    switch(axis) {
        case Axis::X:
            return fscalco * static_cast<float>(traceInfo.gx + traceInfo.sx) * 0.5f;
            break;
        case Axis::Y:
            return fscalco * static_cast<float>(traceInfo.gy + traceInfo.sy) * 0.5f;
            break;
    }

    return 0;
}

float Utils::halfoffset(trace_info_t traceInfo, float azimuth) {
    float hx = Utils::halfoffset_axis(traceInfo, Axis::X),
          hy = Utils::halfoffset_axis(traceInfo, Axis::Y);

    return hx * sin(azimuth) + hy * cos(azimuth);
}

float Utils::midpoint(trace_info_t traceInfo, float azimuth) {
    float mx = Utils::midpoint_axis(traceInfo, Axis::X),
          my = Utils::midpoint_axis(traceInfo, Axis::Y);

    return mx * sin(azimuth) + my * cos(azimuth);
}

float Utils::degreesToRad(float degree) {
    return degree * static_cast<float>(M_PI) / 180.0f;
}

float Utils::radToDegrees(float rad) {
    return rad * 180.0f / static_cast<float>(M_PI);
}
