#pragma once

#include <memory>
#include <sstream>

using namespace std;

enum class LogLevel {
    ERROR,
    INFO,
    LOW,
    HIGH,
    DEBUG
};

class Logger {
    private:
        static unique_ptr<Logger> instance;
        LogLevel verbosity;

    public:
        Logger(LogLevel defaultVerbosity = LogLevel::INFO);
        ~Logger();

        LogLevel getVerbosity() const;
        void setVerbosity(LogLevel newVerbosity);

        static Logger* getInstance();

        static void print(LogLevel lvl, const string& message);
        static void print(LogLevel lvl, ostringstream& stringStream);
};

#define LOGE(m) Logger::print(LogLevel::ERROR, m)
#define LOGI(m) Logger::print(LogLevel::INFO, m)
#define LOGL(m) Logger::print(LogLevel::LOW, m)
#define LOGH(m) Logger::print(LogLevel::HIGH, m)
#define LOGD(m) Logger::print(LogLevel::DEBUG, m)
