#include "cuda/include/execution/CudaSingleHostRunner.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/algorithm/CudaComputeAlgorithmBuilder.hpp"
#include "cuda/include/semblance/data/CudaDeviceContextBuilder.hpp"

#include <cuda.h>
#include <sstream>
#include <stdexcept>

#include <memory>

using namespace std;

CudaSingleHostRunner::CudaSingleHostRunner(Parser* parser) :
    SingleHostRunner(parser, CudaComputeAlgorithmBuilder::getInstance(), CudaDeviceContextBuilder::getInstance()) {
}

unsigned int CudaSingleHostRunner::getNumOfDevices() const {

    int devicesCount;

    CUDA_ASSERT(cudaGetDeviceCount(&devicesCount));

    if (devicesCount == 0) {
        throw runtime_error("No CUDA devices found on your system. Exiting.");
    }

    return static_cast<unsigned int>(devicesCount);
}
