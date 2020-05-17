#include "common/include/parser/LinearSearchParser.hpp"

using namespace boost::program_options;
using namespace std;

unique_ptr<Parser> LinearSearchParser::instance = nullptr;

LinearSearchParser::LinearSearchParser() : Parser() {
    arguments.add_options()
        ("granularity", value<vector<int>>()->required(), "Discretization granularities.");
}

ComputeAlgorithm* LinearSearchParser::parseComputeAlgorithm(
    ComputeAlgorithmBuilder* builder,
    shared_ptr<DeviceContext> deviceContext,
    shared_ptr<Traveltime> traveltime
) const {

    vector<int> discretizationGranularity;

    if (argumentMap.count("granularity")) {
        discretizationGranularity = argumentMap["granularity"].as<vector<int>>();
    }

    return builder->buildLinearSearchAlgorithm(traveltime, deviceContext, discretizationGranularity);
}

Parser* LinearSearchParser::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<LinearSearchParser>();
    }
    return instance.get();
}
