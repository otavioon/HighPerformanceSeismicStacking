#include "cuda/include/execution/CudaSingleHostRunner.hpp"
#include "cuda/include/semblance/algorithm/CudaAlgorithmBuilder.hpp"
#include "cuda/include/semblance/data/CudaDeviceContextBuilder.hpp"

#include <cuda.h>
#include <sstream>
#include <stdexcept>

#include <memory>

using namespace std;

CudaSingleHostRunner::CudaSingleHostRunner(Parser* parser) :
    SingleHostRunner(parser, CudaAlgorithmBuilder::getInstance(), CudaDeviceContextBuilder::getInstance()) {
}

unsigned int CudaSingleHostRunner::getNumOfDevices() const {

    int devicesCount;

    cudaError_t errorCode = cudaGetDeviceCount(&devicesCount);

    if (devicesCount == 0) {
        throw runtime_error("No CUDA devices found on your system. Exiting.");
    }

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA cudaGetDeviceCount failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    return static_cast<unsigned int>(devicesCount);
}
