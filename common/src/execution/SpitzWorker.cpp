#include "common/include/execution/SpitzWorker.hpp"

using namespace std;

SpitzWorker::SpitzWorker(ComputeAlgorithm* computeAlgorithm) : computeAlgorithm(computeAlgorithm) {
}

int SpitzWorker::run(spitz::istream& task, const spitz::pusher& result) {
    spitz::ostream outputStream;

    float m0 = task.read_float();

    computeAlgorithm->computeSemblanceAtGpuForMidpoint(m0);

    const vector<float>& semblanceResults = computeAlgorithm->getComputedResults();

    outputStream.write_data(semblanceResults.data(), semblanceResults.size() * sizeof(float));

    outputStream.write_float(computeAlgorithm->getStatisticalResult(StatisticResult::EFFICIENCY));

    outputStream.write_float(computeAlgorithm->getStatisticalResult(StatisticResult::INTR_PER_SEC));

    result.push(outputStream);

    return 0;
}
