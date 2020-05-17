#ifndef COMMON_STATISTIC_RESULT_HPP
#define COMMON_STATISTIC_RESULT_HPP

#include <string>
#include <unordered_map>

using namespace std;

enum class StatisticResult {
    EFFICIENCY,
    INTR_PER_SEC,
    CNT
};

extern unordered_map<StatisticResult, string> STATISTIC_NAME_MAP;

#endif
