#ifndef CUDA_SINGLE_HOST_H
#define CUDA_SINGLE_HOST_H

#include "common/include/execution/SingleHostRunner.hpp"
#include "common/include/parser/Parser.hpp"

using namespace std;

class CudaSingleHostRunner : public SingleHostRunner {
    public:
        CudaSingleHostRunner(Parser* parser);
        unsigned int getNumOfDevices() const override;
};
#endif
