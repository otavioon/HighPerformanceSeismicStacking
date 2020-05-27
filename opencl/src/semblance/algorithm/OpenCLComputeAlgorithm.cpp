#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithm.hpp"

#include <CL/cl2.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

OpenCLComputeAlgorithm::OpenCLComputeAlgorithm(
    shared_ptr<OpenCLDeviceContext> context,
    const string& path
) : context(context),
    kernelPath(path) {
    compileKernel();
}

void OpenCLComputeAlgorithm::compileKernel() {
    cl_int errorCode;

    vector<string> kernelSources(kernelPath);
    vector<cl::Kernel> kernelArray;

    cl::Program program(context->getContext(), kernelSources, &errorCode);

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating cl::Program failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    errorCode = program.build();

    if (errorCode != CL_SUCCESS) {
        auto errors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context->getDevice(), &errorCode);

        ostringstream stringStream;
        stringStream << "Building cl::Program failed with " << errors;
        throw runtime_error(stringStream.str());
    }

    errorCode = program.createKernels(&kernelArray);

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating OPENCL kernels failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    for(auto kernel : kernelArray) {
        string kernelName = kernel->getInfo<CL_KERNEL_FUNCTION_NAME>(&errorCode);

        if (errorCode != CL_SUCCESS) {
            ostringstream stringStream;
            stringStream << "Fetching CL_KERNEL_FUNCTION_NAME failed with error " << errorCode;
            throw runtime_error(stringStream.str());
        }

        kernels[kernelName] = *kernel;
    }
}
