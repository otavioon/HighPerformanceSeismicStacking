#pragma once

#include "common/include/semblance/data/DataContainer.hpp"
#include "opencl/include/semblance/data/context.hpp"

#include <CL/cl2.hpp>
#include <memory>

#define OPENCL_DEV_PTR(_ptr) dynamic_cast<OpenCLDataContainer>(_ptr)->getOpenClAddress()

using namespace std;

class OpenCLDataContainer : public DataContainer {

    private:
        unique_ptr<cl::Buffer> openClAddress;
        shared_ptr<DeviceContext> context;

    public:
        OpenCLDataContainer(unsigned int elementCount, shared_ptr<DeviceContext> context);
        ~OpenCLDataContainer();

        cl::Buffer* getOpenClAddress() const;

        void allocate() override;

        void copyFrom(const std::vector<float>& sourceArray) override;

        void copyFromWithOffset(const std::vector<float>& sourceArray, unsigned int offset) override;

        void deallocate() override;

        void pasteTo(std::vector<float>& targetArray) override;

        void reset() override;

        void reallocate(unsigned int newElementCount) override;
};
