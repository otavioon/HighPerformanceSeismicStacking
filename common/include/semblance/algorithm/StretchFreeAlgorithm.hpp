#ifndef SEMBL_ALGORITHM_STRETCH_FREE_H
#define SEMBL_ALGORITHM_STRETCH_FREE_H

#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/semblance/data/DataContainer.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

class StretchFreeAlgorithm : public ComputeAlgorithm {
    protected:
        unordered_map<float, unique_ptr<DataContainer>> nonStretchFreeParameters;

        vector<string> parameterFileArray;

        unsigned int getTotalNumberOfParameters() const;

    public:
        StretchFreeAlgorithm(
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            const vector<string>& files
        );

        void computeSemblanceAndParametersForMidpoint(float m0) override;
        unsigned int getParameterArrayStep() const override;
        void setUp() override;
        const string toString() const override;

        void readNonStretchedFreeParameterFromFile();
};
#endif