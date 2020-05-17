#ifndef SEMBL_DEVICE_CONTEXT_HPP
#define SEMBL_DEVICE_CONTEXT_HPP

class DeviceContext {
    protected:
        unsigned int deviceId;

    public:
        DeviceContext(unsigned int devId);
        virtual ~DeviceContext();
        virtual void activate() const = 0;
};
#endif
