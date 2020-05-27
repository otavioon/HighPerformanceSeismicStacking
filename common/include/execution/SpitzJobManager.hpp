#pragma once

#include "common/include/model/Cdp.hpp"
#include "common/include/model/Gather.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <spitz/spitz.hpp>

using namespace std;

class SpitzJobManager : public spitz::job_manager {
    private:
        map<float, Cdp>::const_iterator cdpIterator;
        mutex iteratorMutex;

    public:
        SpitzJobManager();

        bool next_task(const spitz::pusher& task) override;
};
