#include "common/include/parser/de.hpp"
#include "cuda/include/execution/single-host/base.hpp"

using namespace std;

int main(int argc, char *argv[]) {

    CudaSingleHostExecution singleHostExecution(
        DifferentialEvolutionParser::getInstance()
    );

    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
