#ifndef OPENCL_SINGLE_HOST_H
#define OPENCL_SINGLE_HOST_H

#include "common/include/execution/single-host/base.hpp"

using namespace std;

class OpenClSingleHostExecution : public SingleHostExecution {

    public:
        OpenClSingleHostExecution(Parser* parser);

        unsigned int getNumOfDevices() const override;

        ComputeAlgorithm* getComputeAlgorithmForThisDevice() override;

        void setFactoryUp() override;
};
#endif