#pragma once

class DeviceContext {
    protected:
        unsigned int deviceId;

    public:
        DeviceContext(unsigned int devId);
        virtual ~DeviceContext();
        virtual void activate() const = 0;
};
