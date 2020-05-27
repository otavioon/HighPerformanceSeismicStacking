#pragma once

#include <string>
#include <utility>

using namespace std;

class TraveltimeParameter {
    private:
        pair<float, float> minAndMax;
        string description;

    public:
        TraveltimeParameter(const string& d);

        void updateMinimum(float min);

        void updateMaximum(float max);

        string getParameterDescription() const;

        float getMinimum() const;

        float getMaximum() const;
};
