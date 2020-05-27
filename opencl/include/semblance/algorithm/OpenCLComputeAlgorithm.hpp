#pragma once

#include <CL/cl2.hpp>
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

class OpenCLComputeAlgorithm {
    protected:
        shared_ptr<OpenCLDeviceContext> context;

        string kernelPath;

        unordered_map<string, cl::Kernel> kernels;

    public:
        OpenCLComputeAlgorithm(shared_ptr<OpenCLDeviceContext> context, const string& path);

        void compileKernel();
};
