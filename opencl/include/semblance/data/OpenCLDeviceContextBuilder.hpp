#pragma once

#include "common/include/semblance/data/DeviceContextBuilder.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <memory>

using namespace std;

class OpenCLDeviceContextBuilder : public DeviceContextBuilder {
    protected:
        static unique_ptr<DeviceContextBuilder> instance;

    public:
        DeviceContext* build(unsigned int devId) override;

        static DeviceContextBuilder* getInstance();
};
