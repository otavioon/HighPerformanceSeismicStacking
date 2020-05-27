#pragma once

#include "common/include/semblance/data/DeviceContext.hpp"

class CudaDeviceContext : public DeviceContext {
    public:
        CudaDeviceContext(unsigned int devId);
        void activate() const override;
};
