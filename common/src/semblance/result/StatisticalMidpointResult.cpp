#include "common/include/semblance/result/StatisticalMidpointResult.hpp"

float StatisticalMidpointResult::getStatisticalResultForMidpoint(float m0) const {
    if (statisticalMidpointResult.find(m0) != statisticalMidpointResult.end()) {
        return statisticalMidpointResult.at(m0);
    }

    throw invalid_argument("Empty result for given midpoint.");
}

void StatisticalMidpointResult::setStatisticalResultForMidpoint(float m0, float stat) {
    statisticalMidpointResult[m0] = stat;
}
