#ifndef COMMON_SHARED_MIDPOINT_QUEUE_HPP
#define COMMON_SHARED_MIDPOINT_QUEUE_HPP

#include <mutex>
#include <queue>

using namespace std;

class SharedMidpointQueue {
    private:
        queue<float> sharedMidpointQueue;
        mutex queueMutex;

    public:
        bool isEmpty();
        float dequeueMidpoint();
        void enqueueMidpoint(float m0);
};
#endif
