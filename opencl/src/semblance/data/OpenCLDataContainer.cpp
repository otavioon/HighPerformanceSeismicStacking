#include "common/include/output/log.hpp"
#include "opencl/include/semblance/data/container.hpp"

#include <CL/cl2.hpp>
#include <sstream>
#include <stdexcept>

using namespace std;

OpenCLDataContainer::OpenCLDataContainer(
    unsigned int elementCount,
    shared_ptr<DeviceContext> context
) : DataContainer(elementCount, context) {
    allocate();
}

cl::Buffer* OpenCLDataContainer::getOpenClAddress() const {
    return openClAddress.get();
}

void OpenCLDataContainer::allocate() {
    cl_int errorCode;

    shared_ptr<OpenCLDeviceContext> OpenCLDeviceContext = dynamic_pointer_cast<OpenCLDeviceContext>(context)->getContext();

    openClAddress = make_unique<cl::Buffer>(
        OpenCLDeviceContext->getContext(),
        CL_MEM_READ_ONLY,
        elementCount * sizeof(float),
        &errorCode
    );

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating cl::Buffer failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }

    reset();
}

void OpenCLDataContainer::copyFrom(const vector<float>& sourceArray) {
    if (elementCount < sourceArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from source array size.");
    }

    copyFromWithOffset(sourceArray, 0);
}

void OpenCLDataContainer::copyFromWithOffset(const vector<float>& sourceArray, unsigned int offset) {
    cl_int errorCode;

    shared_ptr<OpenCLDeviceContext> OpenCLDeviceContext = dynamic_pointer_cast<OpenCLDeviceContext>(context)->getContext();

    cl::CommandQueue* commandQueue = OpenCLDeviceContext->getCommandQueue();

    errorCode = commandQueue->enqueueWriteBuffer(
        OpenCLDeviceContext->getContext(),
        CL_TRUE,
        offset * sizeof(float),
        sourceArray.size() * sizeof(float),
        sourceArray.data()
    );

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating cl::CommandQueue::enqueueWriteBuffer failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}

void OpenCLDataContainer::deallocate() {
    LOGI("Deallocatiog data container");
    cudaFree(cudaAddress);
}

void OpenCLDataContainer::pasteTo(vector<float>& targetArray) {
    cl_int errorCode;

    shared_ptr<OpenCLDeviceContext> OpenCLDeviceContext = dynamic_pointer_cast<OpenCLDeviceContext>(context)->getContext();

    cl::CommandQueue* commandQueue = OpenCLDeviceContext->getCommandQueue();

    if (elementCount > targetArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from target array size.");
    }

    errorCode = commandQueue->enqueueReadBuffer(
        OpenCLDeviceContext->getContext(),
        CL_TRUE,
        0,
        targetArray.size() * sizeof(float),
        targetArray.data()
    );

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating cl::CommandQueue::enqueueReadBuffer failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}

void OpenCLDataContainer::reset() {
    cl_int errorCode;

    shared_ptr<OpenCLDeviceContext> OpenCLDeviceContext = dynamic_pointer_cast<OpenCLDeviceContext>(context)->getContext();

    cl::CommandQueue* commandQueue = OpenCLDeviceContext->getCommandQueue();

    errorCode = commandQueue->enqueueFillBuffer<float>(
        OpenCLDeviceContext->getContext(),
        0,
        0,
        size * sizeof(float)
    );

    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "Creating cl::CommandQueue::enqueueFillBuffer failed with error " << errorCode;
        throw runtime_error(stringStream.str());
    }
}
