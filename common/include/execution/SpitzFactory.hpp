#pragma once

#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/execution/SpitzJobManager.hpp"
#include "common/include/execution/SpitzWorker.hpp"
#include "common/include/parser/Parser.hpp"
#include "common/include/semblance/data/DeviceContextBuilder.hpp"

#include <memory>
#include <mutex>
#include <spits.hpp>

using namespace std;

class SpitzFactory : public spits::factory {
    protected:
        Parser* parser;

        mutex deviceMutex;

        ComputeAlgorithmBuilder* builder;

        DeviceContextBuilder* deviceBuilder;

        shared_ptr<mutex> taskMutex;

        shared_ptr<Traveltime> traveltime;

        unsigned int deviceCount = 0;

    public:
        SpitzFactory(Parser* p, ComputeAlgorithmBuilder* builder, DeviceContextBuilder* deviceBuilder);

        spits::job_manager *create_job_manager(
            int argc,
            const char *argv[],
            spits::istream& jobinfo,
            spits::metrics& metrics
        ) override;

        spits::committer *create_committer(
            int argc,
            const char *argv[],
            spits::istream& jobinfo,
            spits::metrics& metrics
        ) override;

        spits::worker *create_worker(
            int argc,
            const char *argv[],
            spits::metrics& metrics
        ) override;

        void initialize(int argc, const char *argv[]);
};
