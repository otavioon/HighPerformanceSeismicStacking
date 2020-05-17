#include "cuda/include/semblance/data/CudaDeviceContext.hpp"
#include "cuda/include/semblance/data/CudaDeviceContextBuilder.hpp"

unique_ptr<DeviceContextBuilder> CudaDeviceContextBuilder::instance = nullptr;

DeviceContextBuilder* CudaDeviceContextBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<CudaDeviceContextBuilder>();
    }

    return instance.get();
}

CudaDeviceContext* CudaDeviceContextBuilder::build(unsigned int devId) {
    return new CudaDeviceContext(devId);
}
