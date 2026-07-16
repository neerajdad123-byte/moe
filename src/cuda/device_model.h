// MoEx clean-room. Device (VRAM) residency of model weights.
//
// Uploads the static-resident tensors (embeddings, per-layer attention + norms
// + router) and a chosen working set of expert bundles from the mapped GGUF
// file straight into VRAM via cudaMemcpy. Canonical quantized bytes are kept
// AS-IS in VRAM (no FP16 expansion) — the kernels dequantize inline, matching
// design doc B2. This is the module that turns "0% VRAM" into real residency.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf/gguf.h"
#include "model/manifest.h"

namespace moex {

// A tensor resident in VRAM: raw quantized bytes + shape/type so a kernel can
// decode it. `dptr` is a device pointer.
struct DeviceTensor {
  void* dptr = nullptr;
  uint64_t nbytes = 0;
  gguf::GgmlType type = gguf::GgmlType::UNKNOWN;
  uint64_t rows = 0;  // logical [rows x cols], row-major over cols
  uint64_t cols = 0;
};

// One expert's three resident matrices.
struct DeviceExpert {
  DeviceTensor gate;  // [d_ff x d_model]
  DeviceTensor up;    // [d_ff x d_model]
  DeviceTensor down;  // [d_model x d_ff]
  bool resident = false;
};

// Per-layer resident static tensors.
struct DeviceLayer {
  DeviceTensor attn_norm;
  DeviceTensor attn_q, attn_k, attn_v, attn_output;
  DeviceTensor attn_q_norm, attn_k_norm;
  DeviceTensor ffn_norm;
  DeviceTensor router;  // ffn_gate_inp, F32 [n_experts x d_model]
};

class DeviceModel {
 public:
  ~DeviceModel();

  // Upload static weights for all layers + global tensors. Returns bytes used.
  // `host_base` is the mapped file base pointer (Model::mapped_data()).
  uint64_t upload_static(const gguf::Model& gguf, const Manifest& man,
                         const uint8_t* host_base, std::string* err);

  // Upload experts [layer][0..n_experts) for the layers/experts requested.
  // `which` is a per-layer list of expert ids to make resident. Returns bytes.
  uint64_t upload_experts(const gguf::Model& gguf, const Manifest& man,
                          const uint8_t* host_base,
                          const std::vector<std::vector<uint32_t>>& which,
                          std::string* err);

  const DeviceLayer& layer(uint32_t l) const { return layers_[l]; }
  const DeviceExpert& expert(uint32_t l, uint32_t e) const {
    return experts_[static_cast<size_t>(l) * n_experts_ + e];
  }
  const DeviceTensor& token_embd() const { return token_embd_; }
  const DeviceTensor& output_norm() const { return output_norm_; }
  const DeviceTensor& output() const { return output_; }

  uint64_t bytes_resident() const { return bytes_resident_; }

 private:
  // Allocate + copy one tensor's bytes to VRAM.
  DeviceTensor upload_tensor_(const gguf::TensorInfo* t, const uint8_t* host_base,
                              uint64_t rows, uint64_t cols);
  DeviceTensor upload_slice_(const ByteSlice& s, const uint8_t* host_base);

  std::vector<DeviceLayer> layers_;
  std::vector<DeviceExpert> experts_;
  uint32_t n_experts_ = 0;
  DeviceTensor token_embd_, output_norm_, output_;
  std::vector<void*> allocs_;  // everything we cudaMalloc'd, for cleanup
  uint64_t bytes_resident_ = 0;
};

}  // namespace moex
