#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/model/Trace.hpp"

#define EPSILON 1e-13
#define INDEX_OF(_x) static_cast<unsigned int>(_x)

enum class Axis {
    X,
    Y
};

class Utils {
    public:
        static float halfoffset_axis(trace_info_t traceInfo, Axis axis);
        static float midpoint_axis(trace_info_t traceInfo, Axis axis);

        static float halfoffset(trace_info_t traceInfo, float azimuth);
        static float midpoint(trace_info_t traceInfo, float azimuth);

        static float degreesToRad(float degree);
        static float radToDegrees(float rad);
};
