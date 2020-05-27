#pragma once

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"

#include <memory>

using namespace std;

class OpenCLLinearSearchAlgorithm : public LinearSearchAlgorithm, public OpenCLComputeAlgorithm {
    public:
        OpenCLLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder
        );

        void computeSemblanceAtGpuForMidpoint(float m0) override;
        void selectTracesToBeUsedForMidpoint(float m0) override;
};
