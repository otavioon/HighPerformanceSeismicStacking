#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/data/CudaDataContainerBuilder.hpp"

#include <memory>

using namespace std;

unique_ptr<CudaDataContainerBuilder> CudaDataContainerBuilder::instance = nullptr;

DataContainerBuilder* CudaDataContainerBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<CudaDataContainerBuilder>();
    }
    return instance.get();
}

DataContainer* CudaDataContainerBuilder::build(
    unsigned int allocatedCount,
    shared_ptr<DeviceContext> deviceContext
) {
    return new CudaDataContainer(allocatedCount, deviceContext);
}
