#pragma once

#include <chrono>

#define MEASURE_EXEC_TIME(executionTime, ans) do { \
    chrono::steady_clock::time_point __start = chrono::steady_clock::now(); \
    (ans); \
    executionTime += chrono::steady_clock::now() - __start; \
} while (0);
