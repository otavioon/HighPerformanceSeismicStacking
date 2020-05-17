#ifndef COMMON_TRACE_HPP
#define COMMON_TRACE_HPP

#include <iostream>
#include <vector>
#include <stdint.h>

using namespace std;

typedef struct trace_info {
    int tracl;
    int tracr;
    int fldr;
    int tracf;
    int ep;
    int cdp;
    int cdpt;
    short trid;
    short nvs;
    short nhs;
    short duse;
    int offset;
    int gelev;
    int selev;
    int sdepth;
    int gdel;
    int sdel;
    int swdep;
    int gwdep;
    short scalel;
    short scalco;
    int sx;
    int sy;
    int gx;
    int gy;
    short counit;
    short wevel;
    short swevel;
    short sut;
    short gut;
    short sstat;
    short gstat;
    short tstat;
    short laga;
    short lagb;
    short delrt;
    short muts;
    short mute;
    unsigned short ns;
    unsigned short dt;
    short gain;
    short igc;
    short igi;
    short corr;
    short sfs;
    short sfe;
    short slen;
    short styp;
    short stas;
    short stae;
    short tatyp;
    short afilf;
    short afils;
    short nofilf;
    short nofils;
    short lcf;
    short hcf;
    short lcs;
    short hcs;
    short year;
    short day;
    short hour;
    short minute;
    short sec;
    short timbas;
    short trwf;
    short grnors;
    short grnofr;
    short grnlof;
    short gaps;
    short otrav;
    float d1;
    float f1;
    float d2;
    float f2;
    float ungpow;
    float unscale;
    int ntr;
    short mark;
    short shortpad;
    short unass[14];
} trace_info_t;

class Trace {
    private:
        float halfoffset, midpoint;

        vector<float> samples;

        trace_info_t info;

    public:
        Trace() : halfoffset(0), midpoint(0) {};
        Trace(const vector<float>& sampleData, trace_info_t i, float azimuthInRad);

        float getHalfoffset() const;
        float getMidpoint() const;

        trace_info_t getInfo() const;

        unsigned int getSampleCount() const;
        const vector<float>& getSamples() const;

        bool isMidpointInRange(float lower, float upper);

        void read(ifstream& fin, float azimuthInRad);

        bool operator<(const Trace& other) const;
};
#endif
