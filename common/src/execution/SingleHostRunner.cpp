#include "common/include/execution/SingleHostRunner.hpp"
#include "common/include/output/Dumper.hpp"

#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std;

SingleHostRunner::SingleHostRunner(
    Parser* p,
    ComputeAlgorithmBuilder* algorithmBuilder,
    DeviceContextBuilder* deviceContextBuilder
) : parser(p),
    algorithmBuilder(algorithmBuilder),
    deviceContextBuilder(deviceContextBuilder),
    deviceIndex(0) {
}

ResultSet* SingleHostRunner::getResultSet() {
    return resultSet.get();
}

queue<float>& SingleHostRunner::getMidpointQueue() {
    return midpointQueue;
}

mutex& SingleHostRunner::getDeviceMutex() {
    return deviceMutex;
}

mutex& SingleHostRunner::getResultSetMutex() {
    return resultSetMutex;
}

mutex& SingleHostRunner::getQueueMutex() {
    return queueMutex;
}

ComputeAlgorithm* SingleHostRunner::getComputeAlgorithm() {
    lock_guard<mutex> autoLock(deviceMutex);
    shared_ptr<DeviceContext> devContext(deviceContextBuilder->build(deviceIndex++));
    return parser->parseComputeAlgorithm(algorithmBuilder, devContext, traveltime);
}

void SingleHostRunner::workerThread(SingleHostRunner *ref) {
    float m0;

    unique_ptr<ComputeAlgorithm> computeAlgorithm(ref->getComputeAlgorithm());

    mutex& resultSetMutex = ref->getResultSetMutex();
    mutex& queueMutex = ref->getQueueMutex();

    queue<float>& mipointQueue = ref->getMidpointQueue();

    ResultSet* resultSet = ref->getResultSet();

    computeAlgorithm->setUp();

    while (1) {

        queueMutex.lock();

        if (mipointQueue.empty()) {
            queueMutex.unlock();
            break;
        }

        m0 = mipointQueue.front();
        mipointQueue.pop();

        queueMutex.unlock();

        computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);

        resultSetMutex.lock();

        resultSet->setAllResultsForMidpoint(m0, computeAlgorithm->getComputedResults());

        resultSet->setStatisticalResultForMidpoint(
            m0,
            StatisticResult::EFFICIENCY,
            computeAlgorithm->getStatisticalResult(StatisticResult::EFFICIENCY)
        );

        resultSet->setStatisticalResultForMidpoint(
            m0,
            StatisticResult::INTR_PER_SEC,
            computeAlgorithm->getStatisticalResult(StatisticResult::INTR_PER_SEC)
        );

        resultSetMutex.unlock();
    }
}

int SingleHostRunner::main(int argc, const char *argv[]) {

    try {
        Gather* gather = Gather::getInstance();

        unsigned int devicesCount = getNumOfDevices();

        vector<thread> threads(devicesCount);

        parser->parseArguments(argc, argv);

        traveltime.reset(parser->parseTraveltime());

        parser->readGather();

        for (auto it : gather->getCdps()) {
            midpointQueue.push(it.first);
        }

        resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId] = thread(workerThread, this);
        }

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId].join();
        }

        // std::chrono::duration<double> elpsd_time = std::chrono::steady_clock::now() - st;

        Dumper dumper(parser->getOutputDirectory(), parser->getFilename());

        dumper.createDir();

        dumper.dumpGatherParameters(parser->getFilename());

        dumper.dumpTraveltime(traveltime.get());

        for (unsigned int i = 0; i < traveltime->getNumberOfResults(); i++) {
            dumper.dumpResult(traveltime->getDescriptionForResult(i), resultSet->getArrayForResult(i));
        }

        dumper.dumpStatisticalResult(
            STATISTIC_NAME_MAP[StatisticResult::EFFICIENCY],
            resultSet->get(StatisticResult::EFFICIENCY)
        );

        dumper.dumpStatisticalResult(
            STATISTIC_NAME_MAP[StatisticResult::INTR_PER_SEC],
            resultSet->get(StatisticResult::INTR_PER_SEC)
        );

        return 0;
    }
    catch (const invalid_argument& e) {
        cout << e.what() << endl;
        parser->printHelp();
        return 0;
    }
    catch (const runtime_error& e) {
        cout << e.what() << endl;
        return -ENODEV;
    }
    catch (const exception& e) {
        cout << e.what() << endl;
        return -1;
    }
}
