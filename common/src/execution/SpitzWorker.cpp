#include "common/include/execution/SpitzWorker.hpp"

using namespace std;

SpitzWorker::SpitzWorker(ComputeAlgorithm* computeAlgorithm, shared_ptr<mutex> taskMutex)
: computeAlgorithm(computeAlgorithm),
  taskMutex(taskMutex) {
}

int SpitzWorker::run(spits::istream& task, const spits::pusher& result) {

    spits::ostream outputStream;

    taskMutex->lock();

    float m0 = task.read_float();

    taskMutex->unlock();

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
    }

    taskMutex->lock();

    result.push(outputStream);

    taskMutex->unlock();

    return 0;
}
