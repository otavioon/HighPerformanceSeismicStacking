#include "common/include/parser/de.hpp"
#include "cuda/include/execution/spitz/builder.hpp"

#include <memory>
#include <spitz/spitz.hpp>

using namespace std;

// Creates a builder class.
#define SPITZ_ENTRY_POINT

spitz::builder *spitz_factory = new SemblanceCudaSpitzFactory(
    DifferentialEvolutionParser::getInstance()
);
