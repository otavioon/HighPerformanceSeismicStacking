#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"

#include <cuda.h>
#include <sstream>
#include <stdexcept>

void cudaAssert(cudaError_t errorCode, const char *file, int line) {
    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "CUDA error detected at " << file << "::" << line << " with message " << cudaGetErrorString(errorCode);
        LOGE(stringStream.str());
        throw runtime_error(stringStream.str());
    }
}
