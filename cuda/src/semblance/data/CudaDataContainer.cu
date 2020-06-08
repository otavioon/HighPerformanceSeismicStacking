#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"

#include <cuda.h>
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
    CUDA_ASSERT(cudaMalloc((void **) &cudaAddress, elementCount * sizeof(float)));
    LOGH("Allocating data container for " << elementCount << " elements.");
    reset();
}

void CudaDataContainer::copyFrom(const vector<float>& sourceArray) {
    if (elementCount < sourceArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from source array size.");
    }

    copyFromWithOffset(sourceArray, 0);
}

void CudaDataContainer::copyFromWithOffset(const vector<float>& sourceArray, unsigned int offset) {
    CUDA_ASSERT(cudaMemcpy(
        cudaAddress + offset,
        sourceArray.data(),
        sourceArray.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}

void CudaDataContainer::deallocate() {
    LOGH("Deallocating data container");
    CUDA_ASSERT(cudaFree(cudaAddress));
}

void CudaDataContainer::pasteTo(vector<float>& targetArray) {
    if (elementCount > targetArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from target array size.");
    }

    CUDA_ASSERT(cudaMemcpy(targetArray.data(), cudaAddress, elementCount * sizeof(float), cudaMemcpyDeviceToHost));
}

void CudaDataContainer::reset() {
    CUDA_ASSERT(cudaMemset(cudaAddress, 0, elementCount * sizeof(float)));
}
