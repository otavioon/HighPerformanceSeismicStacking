#pragma once

#include "common/include/semblance/data/DeviceContextBuilder.hpp"
#include "cuda/include/semblance/data/CudaDeviceContext.hpp"

#include <memory>

using namespace std;

class CudaDeviceContextBuilder : public DeviceContextBuilder {
    protected:
        static unique_ptr<DeviceContextBuilder> instance;

    public:
        CudaDeviceContext* build(unsigned int devId) override;

        static DeviceContextBuilder* getInstance();
};
