#include "common/include/output/Logger.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"

#include <cuda.h>
#include <sstream>
#include <stdexcept>

using namespace std;

CudaDataContainer::CudaDataContainer(
    unsigned int elementCount,
    shared_ptr<DeviceContext> context
) : DataContainer(elementCount, context) {
    allocate();
}

CudaDataContainer::~CudaDataContainer() {
    deallocate();
}

float* CudaDataContainer::getCudaAddress() const {
    return cudaAddress;
}

void CudaDataContainer::allocate() {
    cudaError_t errorCode = cudaMalloc((void **) &cudaAddress, elementCount * sizeof(float));

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA buffer failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    reset();
}

void CudaDataContainer::copyFrom(const vector<float>& sourceArray) {
    if (elementCount < sourceArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from source array size.");
    }

    copyFromWithOffset(sourceArray, 0);
}

void CudaDataContainer::copyFromWithOffset(const vector<float>& sourceArray, unsigned int offset) {
    cudaError_t errorCode = cudaMemcpy(
        cudaAddress + offset,
        sourceArray.data(),
        sourceArray.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA memcpy failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}

void CudaDataContainer::deallocate() {
    LOGI("Deallocatiog data container");
    cudaFree(cudaAddress);
}

void CudaDataContainer::pasteTo(vector<float>& targetArray) {
    cudaError_t errorCode;

    if (elementCount > targetArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from target array size.");
    }

    errorCode = cudaMemcpy(targetArray.data(), cudaAddress, elementCount * sizeof(float), cudaMemcpyDeviceToHost);

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA memcpy failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}

void CudaDataContainer::reset() {
    cudaError_t errorCode = cudaMemset(cudaAddress, 0, elementCount * sizeof(float));

    if (errorCode != cudaSuccess) {
        ostringstream stringStream;
        stringStream << "Creating CUDA memset failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}

void CudaDataContainer::reallocate(unsigned int newElementCount) {
    deallocate();
    elementCount = newElementCount;
    allocate();
}
