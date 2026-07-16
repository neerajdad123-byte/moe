// MoEx clean-room. Minimal CUDA error-checking helpers. No ggml/llama code.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace moex {

inline void cuda_check(cudaError_t e, const char* what, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[CUDA] %s failed: %s (%s:%d)\n", what,
                 cudaGetErrorString(e), file, line);
    std::abort();
  }
}

#define MOEX_CUDA(call) ::moex::cuda_check((call), #call, __FILE__, __LINE__)

// Human-readable byte count.
inline std::string human_bytes(uint64_t b) {
  const char* u[] = {"B", "KiB", "MiB", "GiB"};
  double v = static_cast<double>(b);
  int i = 0;
  while (v >= 1024.0 && i < 3) { v /= 1024.0; ++i; }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.3f %s", v, u[i]);
  return buf;
}

}  // namespace moex
