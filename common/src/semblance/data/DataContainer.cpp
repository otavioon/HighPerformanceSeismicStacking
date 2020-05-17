#include "common/include/semblance/data/DataContainer.hpp"

DataContainer::DataContainer(unsigned int count, shared_ptr<DeviceContext> context) :
    deviceContext(context),
    elementCount(count) {
}

DataContainer::~DataContainer() {
}
