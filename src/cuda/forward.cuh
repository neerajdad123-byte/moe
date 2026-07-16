// MoEx clean-room. Batch-1 Qwen3-MoE decode forward pass over VRAM-resident
// weights. Exact router every layer, exact top-8 experts every token.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <vector>

#include "cuda/device_model.h"
#include "cuda/profiler.cuh"
#include "model/manifest.h"

namespace moex {

struct ExpertDispatch;

class Forward {
 public:
  Forward(const DeviceModel& dm, const ModelConfig& cfg, uint32_t max_ctx);
  ~Forward();

  int step(int token_id, int pos, std::vector<int>* route_out);

  std::function<void(uint32_t layer, const int* ids, uint32_t n)> ensure_experts;
  const std::vector<int>& last_route() const { return last_route_; }
  const int* force_route = nullptr;
  StepProf* prof = nullptr;

 private:
  const DeviceModel& dm_;
  ModelConfig cfg_;
  uint32_t max_ctx_;

  float* x_ = nullptr;
  float* xn_ = nullptr;
  float* q_ = nullptr;
  float* k_ = nullptr;
  float* v_ = nullptr;
  float* attn_out_ = nullptr;
  float* tmp_ = nullptr;
  float* router_logits_ = nullptr;
  float* gate_buf_ = nullptr;
  float* up_buf_ = nullptr;
  float* expert_out_ = nullptr;
  float* group_gate_buf_ = nullptr;   // [top_k * d_ff]
  float* group_expert_out_ = nullptr; // [top_k * d_model]
  ExpertDispatch* d_dispatch_ = nullptr;
  ExpertDispatch* h_dispatch_ = nullptr;  // pinned host staging
  float* logits_ = nullptr;
  float* kcache_ = nullptr;
  float* vcache_ = nullptr;

  // GPU reductions
  float* argmax_part_v_ = nullptr;
  int* argmax_part_i_ = nullptr;
  int* d_argmax_ = nullptr;
  int* d_topk_idx_ = nullptr;   // [n_experts_used]
  float* d_topk_w_ = nullptr;   // [n_experts_used]
  int argmax_nblocks_ = 256;

  std::vector<int> last_route_;
  float* h_logits_ = nullptr;
  int* h_topk_idx_ = nullptr;
  float* h_topk_w_ = nullptr;
  float* h_router_ = nullptr;  // only for force_route / debug

  uint64_t kv_stride_layer_() const {
    return (uint64_t)max_ctx_ * cfg_.kv_dim();
  }
};

}  // namespace moex
