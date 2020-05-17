#include "common/include/semblance/result/StatisticResult.hpp"

using namespace std;

unordered_map<StatisticResult, string> STATISTIC_NAME_MAP = {
    { StatisticResult::EFFICIENCY, "efficiency" },
    { StatisticResult::INTR_PER_SEC, "interpolations_per_sec"}
};

