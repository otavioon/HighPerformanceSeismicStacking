#pragma once

#include "common/include/traveltime/Traveltime.hpp"

using namespace std;

class CommonMidPoint : public Traveltime {

    public:
        CommonMidPoint();

        enum traveltime_t getModel() const override;

        const string getTraveltimeWord() const override;

        void updateReferenceHalfoffset(float h0) override;

        const string toString() const override;
};
