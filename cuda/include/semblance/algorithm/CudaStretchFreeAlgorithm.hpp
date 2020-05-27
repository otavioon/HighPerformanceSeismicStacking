#pragma once

#include "common/include/semblance/algorithm/StretchFreeAlgorithm.hpp"

#include <memory>

class CudaStretchFreeAlgorithm : public StretchFreeAlgorithm {

    public:
        CudaStretchFreeAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            const vector<string>& files
        );

        void compileDeviceKernel(const string& kernelSourcePath) override;

        void computeSemblanceAtGpuForMidpoint(float m0) override;

        void selectTracesToBeUsedForMidpoint(float m0) override;
};
