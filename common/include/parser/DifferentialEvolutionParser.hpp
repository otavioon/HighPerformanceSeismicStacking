#ifndef COMMON_DIFF_EVOLUTION_PARSER_HPP
#define COMMON_DIFF_EVOLUTION_PARSER_HPP

#include "common/include/parser/Parser.hpp"

#include <memory>

using namespace std;

class DifferentialEvolutionParser : public Parser {
    protected:
        static unique_ptr<Parser> instance;

    public:
        DifferentialEvolutionParser();

        ComputeAlgorithm* parseComputeAlgorithm(
            ComputeAlgorithmBuilder* builder,
            shared_ptr<DeviceContext> deviceContext,
            shared_ptr<Traveltime> traveltime
        ) const override;

        static Parser* getInstance();
};
#endif
