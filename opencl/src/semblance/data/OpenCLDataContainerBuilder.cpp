#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainerBuilder.hpp"

#include <memory>

using namespace std;

unique_ptr<OpenCLDataContainerBuilder> OpenCLDataContainerBuilder::instance = nullptr;

DataContainerBuilder* OpenCLDataContainerBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<OpenCLDataContainerBuilder>();
    }
    return instance.get();
}

DataContainer* OpenCLDataContainerBuilder::build(
    unsigned int allocatedCount,
    shared_ptr<DeviceContext> deviceContext
) {
    return new OpenCLDataContainer(allocatedCount, deviceContext);
}
