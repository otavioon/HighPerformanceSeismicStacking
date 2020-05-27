#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContextBuilder.hpp"

using namespace std;

unique_ptr<DeviceContextBuilder> OpenCLDeviceContextBuilder::instance = nullptr;

DeviceContextBuilder* OpenCLDeviceContextBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<OpenCLDeviceContextBuilder>();
    }

    return instance.get();
}

DeviceContext* OpenCLDeviceContextBuilder::build(unsigned int devId) {
    return new OpenCLDeviceContext(devId);
}
