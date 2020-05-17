#ifndef SEMBL_CUDA_ALGORITHM_LINEAR_SEARCH_H
#define SEMBL_CUDA_ALGORITHM_LINEAR_SEARCH_H

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"

#include <memory>

class CudaLinearSearchAlgorithm : public LinearSearchAlgorithm {

    public:
        CudaLinearSearchAlgorithm(
            shared_ptr<Traveltime> model,
            DataContainerBuilder* dataBuilder
        );

        void computeSemblanceAtGpuForMidpoint(float m0) override;
        void selectTracesToBeUsedForMidpoint(float m0) override;
};
#endif
