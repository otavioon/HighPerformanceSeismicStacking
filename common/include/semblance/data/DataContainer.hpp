#ifndef SEMBL_DATA_CONTAINER_H
#define SEMBL_DATA_CONTAINER_H

#include "common/include/semblance/data/DeviceContext.hpp"

#include <memory>
#include <vector>

using namespace std;

class DataContainer {
    protected:
        shared_ptr<DeviceContext> deviceContext;
        unsigned int elementCount;

    public:
        DataContainer(unsigned int count, shared_ptr<DeviceContext> context);
        virtual ~DataContainer();

        unsigned int getElementCount() { return elementCount; };

        //
        // Virtual methods.
        //

        virtual void allocate() = 0;

        virtual void deallocate() = 0;

        virtual void copyFrom(const vector<float>& sourceArray) = 0;

        virtual void copyFromWithOffset(const vector<float>& sourceArray, unsigned int offset) = 0;

        virtual void pasteTo(vector<float>& targetArray) = 0;

        virtual void reallocate(unsigned int newElementCount) = 0;

        virtual void reset() = 0;
};

#endif