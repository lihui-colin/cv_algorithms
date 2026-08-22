#pragma once
#include <string>

namespace openshape {

struct CpuFeatures {
  bool avx2 = false;
  bool avx512f = false;
  bool neon = false;
};

CpuFeatures detect_cpu_features();
std::string cpu_features_string();

// Stable runtime dispatch reporting. Unsupported CPUs retain the portable
// SoA implementation; ISA-specific kernels are selected only after probing.
const char* active_score_kernel_name();
const char* active_matcher_score_kernel_name();

}
