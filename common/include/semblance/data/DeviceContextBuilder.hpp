#pragma once

#include "common/include/semblance/data/DeviceContext.hpp"

class DeviceContextBuilder {
    public:
        virtual ~DeviceContextBuilder() {};
        virtual DeviceContext* build(unsigned int devId) = 0;
};
