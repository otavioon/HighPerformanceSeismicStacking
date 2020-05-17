#include "common/include/traveltime/TraveltimeParameter.hpp"

using namespace std;

TraveltimeParameter::TraveltimeParameter(const string& d) : description(d) {
}

void TraveltimeParameter::updateMinimum(float min) {
    minAndMax.first = min;
}

void TraveltimeParameter::updateMaximum(float max) {
    minAndMax.second = max;
}

string TraveltimeParameter::getParameterDescription() const {
    return description;
}

float TraveltimeParameter::getMinimum() const {
    return minAndMax.first;
}

float TraveltimeParameter::getMaximum() const {
    return minAndMax.second;
}
