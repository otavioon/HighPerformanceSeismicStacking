#ifndef COMMON_SEMBL_ALGO_BUILDER_HPP
#define COMMON_SEMBL_ALGO_BUILDER_HPP

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"
#include "common/include/semblance/algorithm/DifferentialEvolutionAlgorithm.hpp"
#include "common/include/semblance/algorithm/StretchFreeAlgorithm.hpp"

#include <memory>

using namespace std;

class ComputeAlgorithmBuilder {
    public:
        virtual LinearSearchAlgorithm* buildLinearSearchAlgorithm(
                shared_ptr<Traveltime> traveltime,
                shared_ptr<DeviceContext> context,
                const vector<int>& discretizationArray
        ) = 0;

        virtual DifferentialEvolutionAlgorithm* buildDifferentialEvolutionAlgorithm(
                shared_ptr<Traveltime> traveltime,
                shared_ptr<DeviceContext> context,
                unsigned int generation,
                unsigned int individualsPerPopulation
        ) = 0;

        virtual StretchFreeAlgorithm* buildStretchFreeAlgorithm(
                shared_ptr<Traveltime> traveltime,
                shared_ptr<DeviceContext> context,
                const vector<string>& parameterFileArray
        ) = 0;
};
#endif
