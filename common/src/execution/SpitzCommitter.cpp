#include "common/include/parser/Parser.hpp"
#include "common/include/output/Dumper.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <memory>
#include <sstream>

using namespace std;

SpitzCommitter::SpitzCommitter(
    shared_ptr<Traveltime> model,
    const string& folder,
    const string& file
) : traveltime(model), folderPath(folder), filePath(file) {

    Gather* gather = Gather::getInstance();

    unsigned int sampleCount = gather->getSamplesPerTrace();

    tempResultArray.resize(traveltime->getNumberOfResults() * sampleCount);

    resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

    taskCount = gather->getTotalCdpsCount();
    taskIndex = 0;

    LOGI("[CO] Committer created.");
}

int SpitzCommitter::commit_task(spitz::istream& result) {

    unique_lock<mutex> mlock(taskMutex);

    float m0 = result.read_float();

    result.read_data(tempResultArray.data(), tempResultArray.size() * sizeof(float));

    resultSet->setAllResultsForMidpoint(m0, tempResultArray);

    resultSet->setStatisticalResultForMidpoint(
        m0,
        StatisticResult::EFFICIENCY,
        result.read_float()
    );

    resultSet->setStatisticalResultForMidpoint(
        m0,
        StatisticResult::INTR_PER_SEC,
        result.read_float()
    );

    taskIndex++;

    LOGI("[CO] Result committed. [" << taskIndex << "," << taskCount << "]" << endl);

    return 0;
}

int SpitzCommitter::commit_job(const spitz::pusher& final_result) {

    //std::chrono::duration<double> diff = std::chrono::steady_clock::now() - strt_tm;

    Dumper dumper(folderPath, filePath);

    dumper.dumpGatherParameters(filePath);

    for (unsigned int i = 0; i < traveltime->getNumberOfResults(); i++) {
        dumper.dumpResult(
            traveltime->getDescriptionForResult(i),
            resultSet->getArrayForResult(i)
        );
    }

    dumper.dumpStatisticalResult(
        STATISTIC_NAME_MAP[StatisticResult::EFFICIENCY],
        resultSet->get(StatisticResult::EFFICIENCY)
    );

    dumper.dumpStatisticalResult(
        STATISTIC_NAME_MAP[StatisticResult::INTR_PER_SEC],
        resultSet->get(StatisticResult::INTR_PER_SEC)
    );

    // A result must be pushed even if the final result is not passed on
    final_result.push(NULL, 0);

    LOGI("[CO] Task completed.");
    //LOGI("[CO] Task completed. It took %.3fs.", diff.count());

    return 0;
}
