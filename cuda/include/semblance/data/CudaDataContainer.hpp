#ifndef CUDA_SEMBL_DATA_CONTAINER_H
#define CUDA_SEMBL_DATA_CONTAINER_H

#include "common/include/semblance/data/DataContainer.hpp"

#include <memory>

#define CUDA_DEV_PTR(_ptr) dynamic_cast<CudaDataContainer*>(_ptr.get())->getCudaAddress()

class CudaDataContainer : public DataContainer {
    private:
        float* cudaAddress;

    public:
        CudaDataContainer(unsigned int elementCount, shared_ptr<DeviceContext> context);
        ~CudaDataContainer();

        float* getCudaAddress() const;

        void allocate() override;

        void copyFrom(const std::vector<float>& sourceArray) override;

        void copyFromWithOffset(const std::vector<float>& sourceArray, unsigned int offset) override;

        void deallocate() override;

        void pasteTo(std::vector<float>& targetArray) override;

        void reset() override;

        void reallocate(unsigned int newElementCount) override;
};
#endif