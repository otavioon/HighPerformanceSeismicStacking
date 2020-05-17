#ifndef SEMBL_CUDA_ALGORITHM_STRETCH_FREE_HPP
#define SEMBL_CUDA_ALGORITHM_STRETCH_FREE_HPP

#include "common/include/semblance/algorithm/stretch_free.hpp"

#include <memory>

class CudaStretchFreeAlgorithm : public StretchFreeAlgorithm {

    public:
        CudaStretchFreeAlgorithm(
            shared_ptr<Traveltime> model,
            DataContainerBuilder* dataBuilder,
            const vector<string>& files
        );

        void computeSemblanceAtGpuForMidpoint(float m0) override;

        void selectTracesToBeUsedForMidpoint(float m0) override;
};
#endif
