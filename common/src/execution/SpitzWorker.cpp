#include "common/include/execution/SpitzWorker.hpp"

using namespace std;

SpitzWorker::SpitzWorker(ComputeAlgorithm* computeAlgorithm) : computeAlgorithm(computeAlgorithm) {
}

int SpitzWorker::run(spitz::istream& task, const spitz::pusher& result) {
    spitz::ostream outputStream;

    float m0 = task.read_float();

    LOGI("m0 = " << m0);

    computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);

    const vector<float>& semblanceResults = computeAlgorithm->getComputedResults();

    outputStream.write_float(m0);

    outputStream.write_data(semblanceResults.data(), semblanceResults.size() * sizeof(float));

    for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
        StatisticResult statResult = static_cast<StatisticResult>(i);
        outputStream.write_float(computeAlgorithm->getStatisticalResult(statResult));
    }

    result.push(outputStream);

    return 0;
}
