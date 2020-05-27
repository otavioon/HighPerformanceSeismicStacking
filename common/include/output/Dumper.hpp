#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/semblance/result/MidpointResult.hpp"
#include "common/include/semblance/result/StatisticalMidpointResult.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <map>
#include <memory>
#include <string>

#define MAX_FILE_DIR_L 256

using namespace std;

class Dumper {
    private:
        string outputDirectoryPath;

    public:
        Dumper(const string& path, const string& dirname);

        void createDir() const;

        void dumpAlgorithm(ComputeAlgorithm* algorithm) const;

        void dumpGatherParameters(const string& file) const;

        void dumpResult(const string& resultName, const MidpointResult& result) const;

        void dumptraveltime(Traveltime* model) const;

        void dumpStatisticalResult(const string& statResultName, const StatisticalMidpointResult& statResult) const;

        //void dumpAll(const SemblanceHostResult& results, float totalElapsedTime) const;
};
