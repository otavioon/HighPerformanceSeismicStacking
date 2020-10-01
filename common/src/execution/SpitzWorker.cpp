#include <chrono>
#include "common/include/execution/SpitzWorker.hpp"

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

    auto start = chrono::high_resolution_clock::now();
    
    computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);
    const vector<float>& semblanceResults = computeAlgorithm->getComputedResults();
    
    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
    metrics.add_metric("semblance_time", time_taken);
    

    outputStream.write_float(m0);

    outputStream.write_data(semblanceResults.data(), semblanceResults.size() * sizeof(float));

    for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
        StatisticResult statResult = static_cast<StatisticResult>(i);
        outputStream.write_float(computeAlgorithm->getStatisticalResult(statResult));
    }

    result.push(outputStream);

    return 0;
}
