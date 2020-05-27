#include "common/include/output/Logger.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

OpenCLDeviceContext::OpenCLDeviceContext(unsigned int deviceId) {

    vector<cl::Platform> platforms;
    vector<cl::Device> devicesPerPlatform;

    // Get platform and device information
    unsigned int numbeOfDevices;
    unsigned int numberOfPlatforms;

    cl_int errorCode = cl::Platform::get(&platforms);

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Getting platform failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    unsigned int deviceCount = 0;
    bool foundDevice = false;
    for (unsigned int platformId = 0; platformId < numberOfPlatforms && !foundDevice; platformId++) {

        errorCode = platforms[platformId].getDevices(CL_DEVICE_TYPE_GPU, &devicesPerPlatform);

        if (errorCode != CL_SUCCESS) {
            ostringstream stringStream;
            stringStream << "Getting devices failed with error " << errorCode;
            throw runtime_error(stringStream.str());
        }

        for (unsigned int deviceId = 0; deviceId < devicesPerPlatform.size(); deviceId++, deviceCount++) {
            if (deviceId == deviceCount) {
                platform = make_unique<cl::Platform>(new cl::Platform(platforms[platformId]));
                device = make_unique<cl::Device>(new cl::Device(devicesPerPlatform[deviceId]));
                foundDevice = true;
                break;
            }
        }
    }

    if (foundDevice) {
        context = make_unique<cl::Context>(*device, &errorCode);

        if (errorCode != CL_SUCCESS) {
            ostringstream stringStream;
            stringStream << "Creating OpenCl's context failed with error " << errorCode;
            throw runtime_error(stringStream.str());
        }

        commandQueue = make_unique<cl::CommandQueue>(*context, *device, &errorCode);

        if (errorCode != CL_SUCCESS) {
            ostringstream stringStream;
            stringStream << "Creating OpenCl's command queue failed with error " << errorCode;
            throw runtime_error(stringStream.str());
        }
    }

    throw runtime_error("Couldn't find device.");
}

OpenCLDeviceContext::OpenCLDeviceContext(const cl::Platform& p, const cl::Device& d) {

    cl_int errorCode;

    platform = make_unique<cl::Platform>(new cl::Platform(p));
    device = make_unique<cl::Device>(new cl::Device(d));

    context = make_unique<cl::Context>(*device, &errorCode);

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating OpenCl's context failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    commandQueue = make_unique<cl::CommandQueue>(*context, *device, &errorCode);

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating OpenCl's command queue failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}

cl::Context* OpenCLDeviceContext::getContext() {
    return context.get();
}

cl::CommandQueue* getCommandQueue() {
    return commandQueue.get();
}

// void OpenCLDeviceContext::compile(std::string pth) {

//     FILE *fp = fopen(pth.append("/semblance.cl").c_str(), "r");

//     uint32_t max_sz = 2 * 1024 * 1024;
//     cl_int ret;

//     char *source = (char*) calloc(max_sz, sizeof(char));
//     size_t source_sz = fread(source, sizeof(char), max_sz, fp);
//     fclose(fp);

//     prgm = clCreateProgramWithSource(ctx, 1, const_cast<const char**>(&source), &source_sz, &ret);

//     ret = clBuildProgram(prgm, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL);

//     if(ret != CL_SUCCESS) {
//         size_t len = 0;

//         clGetProgramBuildInfo(prgm, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

//         char *buffer = (char*) calloc(len, sizeof(char));

//         ret = clGetProgramBuildInfo(prgm, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

//         LOGI("%s", buffer);

//         free(buffer);

//         throw "OpenCL program failed to build!";
//     }

//     sembl_kernel = clCreateKernel(prgm, "compute_semblance_gpu", &ret);
//     sembl_ga_kernel = clCreateKernel(prgm, "compute_semblance_ga_gpu", &ret);
//     stack_kernel = clCreateKernel(prgm, "compute_strech_free_sembl_gpu", &ret);
//     sembl_ft_cmp_crs = clCreateKernel(prgm, "search_for_traces_cmp_crs", &ret);
//     sembl_ft_crp = clCreateKernel(prgm, "search_for_traces_crp", &ret);

//     if(ret != CL_SUCCESS){
//         LOGI("clCreateKernel failed");
//         throw "OpenCL clCreateKernel failed!";
//     }

//     free(source);
// }
