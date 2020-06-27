#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/output/Logger.hpp"

#include <memory>
#include <spitz/spitz.hpp>

using namespace std;

SpitzFactory::SpitzFactory(
    Parser* p,
    ComputeAlgorithmBuilder* builder,
    DeviceContextBuilder* deviceBuilder
) : parser(p), builder(builder), deviceBuilder(deviceBuilder) {
}

void SpitzFactory::initialize(int argc, const char *argv[]) {
    parser->parseArguments(argc, argv);
    parser->readGather();

    if (traveltime == nullptr) {
        LOGH("Initializing traveltime");
        traveltime.reset(parser->parseTraveltime());
    }

    LOGI("Factory initialized");
}

spitz::job_manager *SpitzFactory::create_job_manager(
    int argc,
    const char *argv[],
    spitz::istream& jobinfo
) {
    initialize(argc, argv);
    return new SpitzJobManager();
}

spitz::committer *SpitzFactory::create_committer(
    int argc,
    const char *argv[],
    spitz::istream& jobinfo
) {
    initialize(argc, argv);

    return new SpitzCommitter(traveltime, parser->getOutputDirectory(), parser->getFilename());
}

spitz::worker *SpitzFactory::create_worker(
    int argc,
    const char *argv[]
) {
    unique_lock<mutex> mlock(deviceMutex);

    initialize(argc, argv);

    shared_ptr<DeviceContext> deviceContext(deviceBuilder->build(deviceCount++));

    ComputeAlgorithm* computeAlgorithm = parser->parseComputeAlgorithm(builder, deviceContext, traveltime);

    computeAlgorithm->setUp();

    return new SpitzWorker(computeAlgorithm);
}
