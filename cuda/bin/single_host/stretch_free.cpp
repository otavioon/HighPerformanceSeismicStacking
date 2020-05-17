#include "common/include/parser/stretch_free.hpp"
#include "cuda/include/execution/single-host/base.hpp"

using namespace std;

int main(int argc, char *argv[]) {

    CudaSingleHostExecution singleHostExecution(
        StretchFreeParser::getInstance()
    );

    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
