#include "common/include/semblance/data/DataContainerBuilder.hpp"

DataContainerBuilder::~DataContainerBuilder() {
}

DataContainer* DataContainerBuilder::build(shared_ptr<DeviceContext> deviceContext) {
    return this->build(0, deviceContext);
}