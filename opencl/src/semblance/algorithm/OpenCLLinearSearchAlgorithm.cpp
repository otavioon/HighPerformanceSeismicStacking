#include "opencl/include/semblance/algorithm/OpenCLLinearSearchAlgorithm.hpp"

#include <memory>

using namespace std;

OpenCLLinearSearchAlgorithm::OpenCLLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : LinearSearchAlgorithm(traveltime, context, dataBuilder),
    OpenCLComputeAlgorithm(context, deviceSource) {
}

void OpenCLLinearSearchAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

    Gather* gather = Gather::getInstance();

    gpu_gather_data_t kernelData = gather->getGpuGatherData();

    gpu_traveltime_data_t kerneltraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    cl::Kernel kernelLinearSearch = kernels["kernelLinearSearch"];

    cl_uint argumentIndex = 0;

    kernelLinearSearch.setArg(argumentIndex++, sizeof(), );

    cl::
}

void OpenCLLinearSearchAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

}
