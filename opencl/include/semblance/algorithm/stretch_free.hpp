#pragma once

#include "common/include/semblance/algorithm/stretch_free.hpp"
#include <memory>

class SemblanceOpenClStretchFreeAlgorithm : public StretchFreeAlgorithm {
    protected:
        unique_ptr<OpenCLDeviceContext> context;

    public:
        SemblanceOpenClStretchFreeAlgorithm(
            Traveltime* model,
            DataContainerBuilder* dataBuilder,
            OpenCLDeviceContext* context,
            const vector<string>& files
        );

        void computeSemblanceAtGpuForMidpoint(float m0) override;
        void selectTracesToBeUsedForMidpoint(float m0) override;
};
