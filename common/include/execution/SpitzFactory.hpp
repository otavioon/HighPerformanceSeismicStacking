#ifndef COMMON_SPITZ_FACTORY_HPP
#define COMMON_SPITZ_FACTORY_HPP

#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/execution/SpitzJobManager.hpp"
#include "common/include/execution/SpitzWorker.hpp"
#include "common/include/parser/Parser.hpp"
#include "common/include/semblance/data/DeviceContextBuilder.hpp"

#include <memory>
#include <spitz/spitz.hpp>

using namespace std;

class SpitzFactory : public spitz::factory {
    protected:
        Parser* parser;

        mutex deviceMutex;

        ComputeAlgorithmBuilder* builder;

        DeviceContextBuilder* deviceBuilder;

        shared_ptr<Traveltime> traveltime;

        unsigned int deviceCount = 0;

    public:
        SpitzFactory(Parser* p, ComputeAlgorithmBuilder* builder, DeviceContextBuilder* deviceBuilder);

        spitz::job_manager *create_job_manager(
            int argc,
            const char *argv[],
            spitz::istream& jobinfo
        ) override;

        spitz::committer *create_committer(
            int argc,
            const char *argv[],
            spitz::istream& jobinfo
        ) override;

        spitz::worker *create_worker(
            int argc,
            const char *argv[]
        ) override;

        void initialize(int argc, const char *argv[]);
};
#endif
