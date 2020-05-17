#ifndef SEMBL_CUDA_DEVICE_CONTEXT_HPP
#define SEMBL_CUDA_DEVICE_CONTEXT_HPP

#include "common/include/semblance/data/DeviceContext.hpp"

class CudaDeviceContext : public DeviceContext {
    public:
        CudaDeviceContext(unsigned int devId);
        void activate() const override;
};
#endif
