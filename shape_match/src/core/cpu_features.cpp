#include "openshape/core/cpu_features.hpp"
#include <sstream>

#if defined(__x86_64__) || defined(__i386__)
#  if defined(__GNUC__) || defined(__clang__)
#    define OPENSHAPE_X86_BUILTIN_CPU 1
#  endif
#endif

namespace openshape {

CpuFeatures detect_cpu_features() {
  CpuFeatures features;
#if defined(OPENSHAPE_X86_BUILTIN_CPU)
  features.avx2 = __builtin_cpu_supports("avx2");
  features.avx512f = __builtin_cpu_supports("avx512f");
#elif defined(_M_X64) || defined(_M_IX86)
  // MSVC runtime probing is intentionally kept conservative until the
  // platform-specific kernels are introduced; portable code remains valid.
  features.avx2 = false;
  features.avx512f = false;
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  features.neon = true;
#endif
  return features;
}

std::string cpu_features_string() {
  const CpuFeatures features = detect_cpu_features();
  std::ostringstream out;
  out << "avx2=" << (features.avx2 ? "1" : "0")
      << ",avx512f=" << (features.avx512f ? "1" : "0")
      << ",neon=" << (features.neon ? "1" : "0");
  return out.str();
}

const char* active_score_kernel_name() {
#if defined(OPENSHAPE_HAS_AVX2_KERNEL) && defined(OPENSHAPE_USE_AVX2_BY_DEFAULT)
  if (detect_cpu_features().avx2) return "avx2";
#endif
  return "portable-soa";
}

const char* active_matcher_score_kernel_name() {
  return "portable-rotated-soa";
}

}
