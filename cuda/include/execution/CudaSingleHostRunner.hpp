#ifndef CUDA_SINGLE_HOST_H
#define CUDA_SINGLE_HOST_H

#include "common/include/execution/SingleHostRunner.hpp"

using namespace std;

class CudaSingleHostRunner : public SingleHostRunner {
    public:
        CudaSingleHostExecution(Parser* parser);
        unsigned int getNumOfDevices() const override;
};
#endif
