#define SPITZ_ENTRY_POINT

#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/parser/LinearSearchParser.hpp"
#include "cuda/include/semblance/algorithm/CudaComputeAlgorithmBuilder.hpp"
#include "cuda/include/semblance/data/CudaDeviceContextBuilder.hpp"

#include <memory>
#include <spitz/spitz.hpp>

using namespace std;

spitz::factory *spitz_factory = new SpitzFactory(
    LinearSearchParser::getInstance(),
    CudaComputeAlgorithmBuilder::getInstance(),
    CudaDeviceContextBuilder::getInstance()
);
