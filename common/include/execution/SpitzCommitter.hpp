#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/semblance/result/ResultSet.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <spits.hpp>
#include <string>
#include <vector>

using namespace std;

class SpitzCommitter : public spits::committer {
    private:
        mutex taskMutex;

        shared_ptr<Traveltime> traveltime;

        string folderPath, filePath;

        unique_ptr<ResultSet> resultSet;

        unsigned int taskCount, taskIndex;

        vector<float> tempResultArray;

        chrono::steady_clock::time_point startTimePoint;

    public:
        SpitzCommitter(
            shared_ptr<Traveltime> traveltime,
            const string& folder,
            const string& file
        );

        int commit_task(spits::istream& result);
        int commit_job(const spits::pusher& final_result);
};
