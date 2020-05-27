#pragma once

#include <CL/cl2.hpp>
#include <memory>
#include <string>

using namespace std;

class OpenCLDeviceContext : public DeviceContext{

    private:
        unique_ptr<cl::Platform> platform;
        unique_ptr<cl::Device> device;
        unique_ptr<cl::Context> context;
        unique_ptr<cl::CommandQueue> commandQueue;

    public:
        OpenCLDeviceContext(unsigned int deviceId);
        OpenCLDeviceContext(const cl::Platform& p, const cl::Device& d);

        cl::Context* getContext();
        cl::CommandQueue* getCommandQueue();
        cl::Device* getDevice();
};
