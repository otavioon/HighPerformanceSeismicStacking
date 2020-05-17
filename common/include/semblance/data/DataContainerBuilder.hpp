#ifndef COMMON_SEMBL_DATA_CONTAINER_BUILDER_HPP
#define COMMON_SEMBL_DATA_CONTAINER_BUILDER_HPP

#include "common/include/semblance/data/DataContainer.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"

#include <memory>

using namespace std;

class DataContainerBuilder {
    public:
        virtual ~DataContainerBuilder();
        DataContainer* build(shared_ptr<DeviceContext> deviceContext);
        virtual DataContainer* build(unsigned int allocatedCount, shared_ptr<DeviceContext> deviceContext) = 0;
};
#endif
