#ifndef TRAVELTIME_OFFSET_CONTINUATION_TRAJECTORY_H
#define TRAVELTIME_OFFSET_CONTINUATION_TRAJECTORY_H

#include "common/include/traveltime/Traveltime.hpp"

using namespace std;

class OffsetContinuationTrajectory : public Traveltime {

    public:
        OffsetContinuationTrajectory();

        enum traveltime_t getModel() const override;

        const string getTraveltimeWord() const override;

        void updateReferenceHalfoffset(float h0) override;

        const string toString() const override;
};
#endif
