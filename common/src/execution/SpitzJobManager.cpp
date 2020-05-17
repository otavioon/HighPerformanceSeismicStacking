#include "common/include/parser/Parser.hpp"
#include "common/include/execution/SpitzJobManager.hpp"

#include <iostream>

using namespace std;

SpitzJobManager::SpitzJobManager() : cdpIterator(Gather::getInstance()->getCdps().cbegin()) {
    cout << "[JM] Job manager created." << endl;
}

bool SpitzJobManager::next_task(const spitz::pusher& task) {
    unique_lock<mutex> mlock(iteratorMutex);

    Gather* gather = Gather::getInstance();

    if (cdpIterator != gather->getCdps().end()) {

        spitz::ostream taskStream;

        taskStream << cdpIterator->first;

        task.push(taskStream);

        cdpIterator++;

        return true;
    }

    return false;
}
