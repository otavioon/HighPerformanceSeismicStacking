#include "common/include/parser/StretchFreeParser.hpp"

using namespace boost::program_options;
using namespace std;

unique_ptr<Parser> StretchFreeParser::instance = nullptr;

StretchFreeParser::StretchFreeParser() : Parser() {
    arguments.add_options()
        ("stack-datafiles", value<string>()->required(), "Datafiles to be used for stretch-free stack generation.");
}

ComputeAlgorithm* StretchFreeParser::parseComputeAlgorithm(
    ComputeAlgorithmBuilder* builder,
    shared_ptr<DeviceContext> deviceContext,
    shared_ptr<Traveltime> traveltime
) const {
    vector<string> nonStretchFreeParameterFiles;

    if (argumentMap.count("stack-datafiles")) {
        nonStretchFreeParameterFiles = argumentMap["stack-datafiles"].as<vector<string>>();
    }

    return builder->buildStretchFreeAlgorithm(traveltime, deviceContext, nonStretchFreeParameterFiles);
}

Parser* StretchFreeParser::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<StretchFreeParser>();
    }
    return instance.get();
}
