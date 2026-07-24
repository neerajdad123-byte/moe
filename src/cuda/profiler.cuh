// MoEx clean-room. Per-token decode profiler.
//
// Threads through Forward::step() and records, without changing what the
// kernels compute:
//   * GPU phase time via CUDA events (embed / attn / router / experts / final)
//   * host sync-point time via CPU clock (router D2H + top-k, logits D2H + argmax)
//   * transfer bytes/calls (H2D, D2H, D2D)
//   * kernel launch count (per type)
//   * bytes dequantized (inline in the GEMVs) for effective-GB/s
//   * expert usage histogram + hit/miss + eviction counts
//
// Two-run discipline: the headline tok/s comes from a CLEAN run with the
// profiler OFF (zero added syncs). The breakdown comes from a separate
// instrumented run. We never quote a number produced while instrumenting.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace moex {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

// Kernel-launch tally by op name.
struct LaunchTally {
  std::map<std::string, long> calls;
  long total = 0;
  void hit(const char* name, int n = 1) {
    calls[name] += n;
    total += n;
  }
  void reset() { calls.clear(); total = 0; }
};

// A pool of CUDA events recorded at phase boundaries within one token, then
// resolved to per-phase milliseconds after a single end-of-token sync.
struct EventPool {
  std::vector<cudaEvent_t> ev;
  int used = 0;
  void ensure(int n) {
    while ((int)ev.size() < n) {
      cudaEvent_t e;
      cudaEventCreate(&e);
      ev.push_back(e);
    }
  }
  void reset() { used = 0; }
  // Record the next boundary; returns its index.
  int mark() {
    ensure(used + 1);
    cudaEventRecord(ev[used]);
    return used++;
  }
  // Elapsed ms between two recorded boundaries (call after sync).
  float span(int a, int b) {
    float m = 0.0f;
    cudaEventElapsedTime(&m, ev[a], ev[b]);
    return m;
  }
  ~EventPool() {
    for (auto e : ev) cudaEventDestroy(e);
  }
};

struct TokenTrace {
  int pos = 0;
  int input_token = 0;
  double wall_ms = 0;
  float embed_gpu_ms = 0, attention_gpu_ms = 0, router_gpu_ms = 0;
  float experts_gpu_ms = 0, final_gpu_ms = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0, vram_used = 0;
};

struct StepProf {
  bool on = false;

  // --- GPU phase accumulators (ms, summed over the timed run) ---
  double embed_ms = 0, attn_ms = 0, router_ms = 0, experts_ms = 0, final_ms = 0;

  // --- Deep sub-phase breakdown (ms). Profile-only; sync-inflated like phases.
  // Attention internals:
  double attn_norm_ms = 0, q_proj_ms = 0, k_proj_ms = 0, v_proj_ms = 0;
  double rope_ms = 0, kv_store_ms = 0, gqa_ms = 0, attn_out_ms = 0;
  double quantize_attn_ms = 0;
  // MoE internals:
  double ffn_norm_ms = 0, router_gemv_ms = 0, router_topk_ms = 0;
  double quantize_model_ms = 0, gate_up_ms = 0, quantize_ff_ms = 0;
  double down_ms = 0, residual_ms = 0;
  // Final internals:
  double final_norm_ms = 0, logits_ms = 0, sample_ms = 0;

  // --- host sync-point accumulators (ms) ---
  double router_sync_ms = 0;  // router logits D2H + host softmax/top-k
  double logits_sync_ms = 0;  // logits D2H + argmax + final device sync

  // --- transfer accounting ---
  uint64_t d2h_bytes = 0;  int d2h_calls = 0;
  uint64_t h2d_bytes = 0;  int h2d_calls = 0;
  uint64_t d2d_bytes = 0;  int d2d_calls = 0;

  // --- launches ---
  LaunchTally launches;

  // --- dequant (inline in GEMV) ---
  uint64_t dequant_bytes_in = 0;   // quantized bytes read + decoded
  uint64_t dequant_elems_out = 0;  // f32 elements produced

  // --- per-token wall latency samples (ms) ---
  std::vector<double> tok_ms;
  std::vector<TokenTrace> trace;

  // --- expert usage histogram (layer-major: layer*n_experts + e) ---
  std::vector<long> expert_hist;
  long expert_hits = 0;    // routed expert already resident (HOT)
  long expert_miss = 0;    // routed expert not resident at dispatch
  long evictions = 0;      // set by the pager (0 in the resident ceiling)

  // --- VRAM ---
  uint64_t vram_peak_used = 0;

  // event pool + per-token boundary bookkeeping
  EventPool pool;
  Clock::time_point tp;  // running host timestamp for coarse phase timing

  void reset_run() {
    embed_ms = attn_ms = router_ms = experts_ms = final_ms = 0;
    attn_norm_ms = q_proj_ms = k_proj_ms = v_proj_ms = 0;
    rope_ms = kv_store_ms = gqa_ms = attn_out_ms = quantize_attn_ms = 0;
    ffn_norm_ms = router_gemv_ms = router_topk_ms = 0;
    quantize_model_ms = gate_up_ms = quantize_ff_ms = 0;
    down_ms = residual_ms = 0;
    final_norm_ms = logits_ms = sample_ms = 0;
    router_sync_ms = logits_sync_ms = 0;
    d2h_bytes = h2d_bytes = d2d_bytes = 0;
    d2h_calls = h2d_calls = d2d_calls = 0;
    launches.reset();
    dequant_bytes_in = dequant_elems_out = 0;
    tok_ms.clear();
    trace.clear();
    for (auto& c : expert_hist) c = 0;
    expert_hits = expert_miss = evictions = 0;
    vram_peak_used = 0;
  }

  void sample_vram() {
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
      uint64_t used = (uint64_t)(t - f);
      if (used > vram_peak_used) vram_peak_used = used;
    }
  }
};

// percentile over a copy-sorted vector
inline double pct(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double idx = p / 100.0 * (v.size() - 1);
  size_t lo = (size_t)idx;
  double frac = idx - lo;
  if (lo + 1 < v.size()) return v[lo] * (1 - frac) + v[lo + 1] * frac;
  return v[lo];
}

}  // namespace moex
