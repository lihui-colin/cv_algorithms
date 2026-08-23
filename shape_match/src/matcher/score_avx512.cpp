// Translation unit reserved for the AVX-512 precise lane kernel.  Keeping it
// separate prevents ISA instructions from entering the portable object.  The
// current exhaustive scorer uses the certified scalar implementation until a
// platform-specific end-to-end benchmark promotes this kernel to Auto.
#include "openshape/core/cpu_features.hpp"

namespace openshape {
const char* avx512_precise_kernel_build_name() { return "avx512"; }
}
