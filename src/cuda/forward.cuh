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
struct RouteTopK;

class Forward {
 public:
  // `man` is needed only for per-layer expert tensor type/row-bytes, which
  // must be knowable without any specific expert being resident (the
  // gpu_dispatch fast path never learns *which* expert id was routed on the
  // host, so it can't derive this from a live DeviceExpert the way the
  // normal path does). Every expert of a layer shares identical geometry
  // (stacked tensor), so this is a one-time, residency-independent read of
  // the manifest at construction, cached per layer.
  Forward(const DeviceModel& dm, const ModelConfig& cfg, uint32_t max_ctx,
         const Manifest& man);
  ~Forward();

  // `weight_out`, if given, mirrors `route_out` layout (n_layers*n_experts_used)
  // and receives each routed expert's renormalized router weight.
  int step(int token_id, int pos, std::vector<int>* route_out,
           std::vector<float>* weight_out = nullptr);

  std::function<void(uint32_t layer, const int* ids, uint32_t n)> ensure_experts;
  const std::vector<int>& last_route() const { return last_route_; }
  const int* force_route = nullptr;
  StepProf* prof = nullptr;

  // Zero-host-round-trip decode: router top-k, expert pointer resolution and
  // dispatch construction all stay on GPU (dispatch_gather_kernel reads
  // DeviceModel's GPU pointer table directly) — no per-layer D2H, no
  // ensure_experts call, no route_out/weight_out population. Caller's
  // contract: every expert this run's routes will ever select must already
  // be resident and stay resident for the run's duration (no concurrent
  // eviction/upload) — exactly the existing "pinned, H2D disabled"
  // ceiling-test invariant. Hit/miss are tracked in GPU counters, read back
  // once via read_hit_miss_counters(), not per token. Only takes effect when
  // force_route == nullptr and prof == nullptr (profiling needs host-visible
  // per-layer route data; force_route is the deterministic-replay debug path).
  bool gpu_dispatch = false;
  void read_hit_miss_counters(unsigned long long* hits, unsigned long long* misses) const;
  void reset_hit_miss_counters();

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
  RouteTopK* d_route_ = nullptr;  // GPU router top-8 output (64 B)
  RouteTopK* h_route_ = nullptr;  // pinned host mirror

  // gpu_dispatch fast-path state
  ExpertDispatch* d_gdispatch_ = nullptr;      // GPU-built dispatch struct
  unsigned long long* d_hit_ctr_ = nullptr;
  unsigned long long* d_miss_ctr_ = nullptr;
  std::vector<int> gu_type_by_layer_, dn_type_by_layer_;
  std::vector<int> gu_rowbytes_by_layer_, dn_rowbytes_by_layer_;

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
