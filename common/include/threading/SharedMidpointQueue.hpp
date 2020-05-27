#pragma once

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
