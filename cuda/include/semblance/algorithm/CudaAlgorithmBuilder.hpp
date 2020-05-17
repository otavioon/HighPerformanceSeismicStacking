#ifndef SEMBL_CUDA_FACTORY_H
#define SEMBL_CUDA_FACTORY_H

#include "common/include/semblance/algorithm/ComputeAlgorithmBuilder.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"

#include <memory>

using namespace std;

class CudaAlgorithmBuilder : public ComputeAlgorithmBuilder {
    protected:
        static unique_ptr<Parser> instance;
    public:
        static const ComputeAlgorithmBuilder* getInstance();

        LinearSearchAlgorithm* buildLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            const vector<int>& discretizationArray
        ) const override;

        DifferentialEvolutionAlgorithm* buildDifferentialEvolutionAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            unsigned int generation,
            unsigned int individualsPerPopulation
        ) const override;

        StretchFreeAlgorithm* buildStretchFreeAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            const vector<string>& parameterFileArray
        ) const override;
};
#endif