#ifndef COMMON_SEMBL_DEV_CONTEXT_BUILDER_HPP
#define COMMON_SEMBL_DEV_CONTEXT_BUILDER_HPP

#include "common/include/semblance/data/DeviceContext.hpp"

class DeviceContextBuilder {
    public:
        virtual ~DeviceContextBuilder() {};
        virtual DeviceContext* build(unsigned int devId) = 0;
};
#endif
