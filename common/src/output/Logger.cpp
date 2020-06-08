#include "common/include/output/Logger.hpp"

#include <stdarg.h>
#include <iostream>

using namespace std;

unique_ptr<Logger> Logger::instance = nullptr;

Logger::Logger(LogLevel defaultVerbosity) : verbosity(defaultVerbosity) {
}

Logger::~Logger() {
}

Logger* Logger::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<Logger>();
    }
    return instance.get();
}

LogLevel Logger::getVerbosity() const {
    return verbosity;
}

void Logger::setVerbosity(LogLevel newVerbosity) {
    verbosity = newVerbosity;
}

void Logger::print(LogLevel lvl, const string& message) {
    if (lvl > getInstance()->getVerbosity())
        return;

    switch(lvl) {
        case LogLevel::ERROR:
            cout << "ERROR: " << message << endl;
            break;
        case LogLevel::INFO:
            cout << " INFO: " << message << endl;
            break;
        case LogLevel::LOW:
            cout << "  LOW: " << message << endl;
            break;
        case LogLevel::HIGH:
            cout << " HIGH: " << message << endl;
            break;
        case LogLevel::DEBUG:
        default:
            cout << "DEBUG: " << message << endl;
    }

    cout.flush();
}

void Logger::print(LogLevel lvl, ostringstream& stringStream) {
    print(lvl, stringStream.str());
    stringStream.str("");
}
