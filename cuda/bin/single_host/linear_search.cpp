#include "common/include/parser/linear_search.hpp"
#include "cuda/include/execution/single-host/base.hpp"

using namespace std;

int main(int argc, char *argv[]) {

    CudaSingleHostExecution singleHostExecution(
        LinearSearchParser::getInstance()
    );

    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
