#include <chrono>
#include "common/include/execution/SpitzWorker.hpp"
#include "common/include/semblance/result/StatisticResult.hpp"

using namespace std;

SpitzWorker::SpitzWorker(ComputeAlgorithm* computeAlgorithm, spits::metrics& metrics)
: computeAlgorithm(computeAlgorithm), metrics(metrics) {
}

int SpitzWorker::run(spits::istream& task, const spits::pusher& result) {

    spits::ostream outputStream;

    float m0 = task.read_float();

    LOGI("m0 = " << m0);

    if (!computeAlgorithm->isSetUp()) {
        computeAlgorithm->setUp();
    }

    computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);

    const vector<float>& semblanceResults = computeAlgorithm->getComputedResults();
    outputStream.write_float(m0);
    outputStream.write_data(semblanceResults.data(), semblanceResults.size() * sizeof(float));

    for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
        StatisticResult statResult = static_cast<StatisticResult>(i);
        outputStream.write_float(computeAlgorithm->getStatisticalResult(statResult));
        metrics.add_metric(STATISTIC_NAME_MAP[statResult], computeAlgorithm->getStatisticalResult(statResult));
    }

    result.push(outputStream);

    return 0;
}
