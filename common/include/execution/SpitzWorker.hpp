#ifndef COMMON_SPITZ_WORKER_HPP
#define COMMON_SPITZ_WORKER_HPP

#include "common/include/parser/Parser.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithmBuilder.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <memory>
#include <spitz/spitz.hpp>

using namespace std;

class SpitzWorker : public spitz::worker {
    protected:
        unique_ptr<ComputeAlgorithm> computeAlgorithm;

    public:
        SpitzWorker(ComputeAlgorithm* computeAlgorithm);
        int run(spitz::istream& task, const spitz::pusher& result);
};
#endif
