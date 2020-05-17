#include "cuda/include/semblance/data/CudaDeviceContext.hpp"

#include <cuda.h>
#include <sstream>
#include <stdexcept>

CudaDeviceContext::CudaDeviceContext(unsigned int devId) : DeviceContext(devId) {
}

void CudaDeviceContext::activate() const {
    cudaError_t errorCode = cudaSetDevice(devId);

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA cudaSetDevice failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}