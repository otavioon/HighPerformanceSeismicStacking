#pragma once

#include "common/include/traveltime/Traveltime.hpp"

#include <memory>

using namespace std;

class TraveltimeBuilder {
    private:
        static unique_ptr<TraveltimeBuilder> instance;

    public:
        static TraveltimeBuilder* getInstance();

        Traveltime* buildCommonMidPoint(
            const vector<float>& lowerBounds,
            const vector<float>& upperBounds
        );

        Traveltime* buildZeroOffsetCommonReflectionSurface(
            const vector<float>& lowerBounds,
            const vector<float>& upperBounds,
            float v0,
            float aIn,
            float bPctg
        );

        Traveltime* buildOffsetContinuationTrajectory(
            const vector<float>& lowerBounds,
            const vector<float>& upperBounds,
            float h0
        );
};
