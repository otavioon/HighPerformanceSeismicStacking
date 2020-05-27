#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/semblance/result/ResultSet.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <spitz/spitz.hpp>
#include <string>
#include <vector>

using namespace std;

class SpitzCommitter : public spitz::committer {
    private:
        mutex taskMutex;

        shared_ptr<Traveltime> traveltime;

        string folderPath, filePath;

        unique_ptr<ResultSet> resultSet;

        unsigned int taskCount, taskIndex;

        vector<float> tempResultArray;

    public:
        SpitzCommitter(
            shared_ptr<Traveltime> traveltime,
            const string& folder,
            const string& file
        );

        int commit_task(spitz::istream& result);
        int commit_job(const spitz::pusher& final_result);
};
