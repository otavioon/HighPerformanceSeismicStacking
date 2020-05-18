#include "common/include/parser/DifferentialEvolutionParser.hpp"
#include "cuda/include/execution/CudaSingleHostRunner.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    CudaSingleHostRunner singleHostExecution(DifferentialEvolutionParser::getInstance());
    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
