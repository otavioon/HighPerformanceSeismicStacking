#include "common/include/model/Trace.hpp"
#include "common/include/model/Utils.hpp"
#include "common/include/output/Logger.hpp"

#include <algorithm>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

Trace::Trace(const vector<float>& sampleData, trace_info_t i, float azimuthInRad) : info(i) {
    midpoint = Utils::midpoint(info, azimuthInRad);
    halfoffset = Utils::halfoffset(info, azimuthInRad);
    samples = sampleData;
}

float Trace::getHalfoffset() const {
    return halfoffset;
}

trace_info_t Trace::getInfo() const {
    return info;
}

float Trace::getMidpoint() const {
    return midpoint;
}

unsigned int Trace::getSampleCount() const {
    return static_cast<unsigned int>(samples.size());
}

const vector<float>& Trace::getSamples() const {
    return samples;
}

bool Trace::isMidpointInRange(float lower, float upper) {
    return midpoint >= lower && midpoint <= upper;
}

void Trace::read(ifstream& fin, float azimuthInRad) {

    if (fin.read(reinterpret_cast<char*>(&info), sizeof(trace_info_t)).eof()) {
        return;
    }

    midpoint = Utils::midpoint(info, azimuthInRad);
    halfoffset = Utils::halfoffset(info, azimuthInRad);

    samples.resize(info.ns);

    LOGD("Reading " << samples.size() * sizeof(float) << " bytes of data");

    if (fin.read(reinterpret_cast<char*>(samples.data()), samples.size() * sizeof(float)).eof()) {
        throw runtime_error("end of file reached before reading all samples.");
    }
}

bool Trace::operator<(const Trace& other) const {

    if (midpoint < other.midpoint) {
        return true;
    }

    if (midpoint == other.midpoint && halfoffset < other.halfoffset) {
        return true;
    }

    return false;
}
