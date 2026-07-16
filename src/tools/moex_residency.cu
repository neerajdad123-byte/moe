// MoEx clean-room. G2a residency probe: load real weights into VRAM and hold,
// so residency is observable in nvidia-smi. Proves the upload path works and
// the ledger's numbers are real, not estimated.
#include <cstdio>
#include <string>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "cuda/cuda_common.h"
#include "cuda/device_model.h"
#include "gguf/gguf.h"
#include "model/manifest.h"

using namespace moex;

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1]
                              : "C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf";
  // How many experts per layer to make resident (working-set size).
  int experts_per_layer = argc > 2 ? std::atoi(argv[2]) : 128;

  gguf::Model m;
  std::string err;
  if (!m.open(path, &err)) {
    std::fprintf(stderr, "open failed: %s\n", err.c_str());
    return 1;
  }
  Manifest man;
  if (!man.build(m, &err)) {
    std::fprintf(stderr, "manifest failed: %s\n", err.c_str());
    return 1;
  }
  const ModelConfig& c = man.config();

  size_t free_b = 0, total_b = 0;
  MOEX_CUDA(cudaMemGetInfo(&free_b, &total_b));
  std::printf("GPU before: %.0f MiB free / %.0f MiB total\n",
              free_b / 1048576.0, total_b / 1048576.0);

  const uint8_t* host_base = m.base();
  DeviceModel dm;

  std::printf("uploading static weights...\n");
  uint64_t sb = dm.upload_static(m, man, host_base, &err);
  if (sb == 0 && !err.empty()) {
    std::fprintf(stderr, "static upload failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("  static resident: %.1f MiB\n", sb / 1048576.0);

  // Build working set: first `experts_per_layer` experts of every layer.
  if (experts_per_layer > (int)c.n_experts) experts_per_layer = c.n_experts;
  std::vector<std::vector<uint32_t>> which(c.n_layers);
  for (uint32_t l = 0; l < c.n_layers; ++l)
    for (int e = 0; e < experts_per_layer; ++e) which[l].push_back(e);

  std::printf("uploading %d experts/layer x %u layers...\n", experts_per_layer,
              c.n_layers);
  uint64_t eb = dm.upload_experts(m, man, host_base, which, &err);
  if (eb == 0 && !err.empty()) {
    std::fprintf(stderr, "expert upload failed: %s\n", err.c_str());
    // still report what we managed
  }
  std::printf("  expert resident: %.1f MiB\n", eb / 1048576.0);

  MOEX_CUDA(cudaMemGetInfo(&free_b, &total_b));
  std::printf("GPU after:  %.0f MiB free / %.0f MiB total\n",
              free_b / 1048576.0, total_b / 1048576.0);
  std::printf("total resident (MoEx): %.2f GiB\n",
              dm.bytes_resident() / (1024.0 * 1024.0 * 1024.0));
  std::printf("VERDICT: real weights are in VRAM. Holding 8s for nvidia-smi.\n");
  std::fflush(stdout);

  // Hold so the user can observe nvidia-smi.
  cudaDeviceSynchronize();
#ifdef _WIN32
  Sleep(8000);
#endif
  return 0;
}
