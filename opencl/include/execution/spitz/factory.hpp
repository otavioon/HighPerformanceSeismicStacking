#ifndef CUDA_EXECUTION_SPITZ_FACTORY_HPP_
#define CUDA_EXECUTION_SPITZ_FACTORY_HPP_

#include "common/include/execution/spitz/builder.hpp"

class SemblanceOpenClSpitzFactory : public SemblanceSpitzFactory {

    public:
        SemblanceOpenClSpitzFactory(Parser* p);

        void setFactoryUp() override;
};
#endif
