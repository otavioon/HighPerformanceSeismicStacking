#ifndef COMMON_STATISTIC_RESULT_MAP_HPP
#define COMMON_STATISTIC_RESULT_MAP_HPP

#include <map>

using namespace std;

class StatisticalMidpointResult {
    private:
        map<float, float> statisticalMidpointResult;
    public:
        float getStatisticalResultForMidpoint(float m0) const;
        void setStatisticalResultForMidpoint(float m0, float stat);
};
#endif
