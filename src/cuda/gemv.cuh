// MoEx clean-room. Quantized GEMV with shared-memory activation cache.
//
// y[rows] = W[rows, cols] . x[cols], W in GGUF blocks, dequant fused into MAC.
// One warp per output row. x is staged once into shared memory per block so
// every row in the block reuses L1-friendly smem instead of re-reading global x
// (the dominant win for batch-1 decode bandwidth).
// The expert path fuses gate+up+SiLU and down+residual. This turns five
// launches per selected expert into two while retaining execution order.
#pragma once

#include "cuda/kernels.cuh"

namespace moex {

__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// Dot one output row with x[cols] already in a device pointer (smem or global).
// K-quant paths are lane-cooperative: all 32 lanes work on each superblock
// together (previously one superblock per lane, idling 24-29 of 32 lanes on
// this model's 8- and 3-superblock rows and leaving q-byte reads uncoalesced).
__device__ __forceinline__ float row_dot(int type, const uint8_t* row,
                                          const float* x, int cols, int lane) {
  float acc = 0.0f;
  if (type == GT_Q4_K) {
    const int nsb = cols / 256;
    for (int sb = 0; sb < nsb; ++sb)
      acc += dot_sb_q4k_lane(row + (size_t)sb * BQ4_K, x + sb * 256, lane);
  } else if (type == GT_Q5_K) {
    const int nsb = cols / 256;
    for (int sb = 0; sb < nsb; ++sb)
      acc += dot_sb_q5k_lane(row + (size_t)sb * BQ5_K, x + sb * 256, lane);
  } else if (type == GT_Q6_K) {
    const int nsb = cols / 256;
    for (int sb = 0; sb < nsb; ++sb)
      acc += dot_sb_q6k_lane(row + (size_t)sb * BQ6_K, x + sb * 256, lane);
  } else if (type == GT_Q8_0) {
    const int nb = cols / 32;
    for (int b = lane; b < nb; b += 32) {
      const uint8_t* p = row + (size_t)b * BQ8_0;
      const float d = rd_h(p);
      const int8_t* q = (const int8_t*)(p + 2);
      const float* xx = x + b * 32;
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < 32; ++i) s += (float)q[i] * xx[i];
      acc += d * s;
    }
  } else if (type == GT_F32) {
    const float* w = (const float*)row;
    for (int i = lane; i < cols; i += 32) acc += w[i] * x[i];
  }
  return warp_reduce_sum(acc);
}

// Stage x into shared memory, then one warp per output row.
// Dynamic smem: cols * sizeof(float). Caller must pass correct 3rd config arg.
__global__ void gemv_q(int type, const uint8_t* W, const float* x, float* y,
                       int rows, int cols, int row_bytes) {
  extern __shared__ float xs[];
  // Cooperative load of activation vector (once per block).
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();

  const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (warp >= rows) return;
  const uint8_t* row = W + (size_t)warp * row_bytes;
  const float r = row_dot(type, row, xs, cols, lane);
  if (lane == 0) y[warp] = r;
}

// One grid covers Q, K and V projections. Each block retains the same
// warp-per-row Q4_K computation as gemv_q, but removes two CPU/driver launches
// per transformer layer.
__global__ void qkv_q_kernel(
    int q_type, const uint8_t* Wq, int q_row_bytes, int k_type,
    const uint8_t* Wk, int k_row_bytes, int v_type, const uint8_t* Wv,
    int v_row_bytes, const float* x, float* q, float* k, float* v,
    int q_rows, int kv_rows, int cols) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row < q_rows) {
    const float out = row_dot(q_type, Wq + (size_t)row * q_row_bytes, xs, cols, lane);
    if (lane == 0) q[row] = out;
  } else if (row < q_rows + kv_rows) {
    const int r = row - q_rows;
    const float out = row_dot(k_type, Wk + (size_t)r * k_row_bytes, xs, cols, lane);
    if (lane == 0) k[r] = out;
  } else if (row < q_rows + 2 * kv_rows) {
    const int r = row - q_rows - kv_rows;
    const float out = row_dot(v_type, Wv + (size_t)r * v_row_bytes, xs, cols, lane);
    if (lane == 0) v[r] = out;
  }
}

inline void launch_qkv_q(int q_type, const uint8_t* Wq, int q_row_bytes,
                         int k_type, const uint8_t* Wk, int k_row_bytes,
                         int v_type, const uint8_t* Wv, int v_row_bytes,
                         const float* x, float* q, float* k, float* v,
                         int q_rows, int kv_rows, int cols) {
  constexpr int threads = 256;
  const int total_rows = q_rows + 2 * kv_rows;
  qkv_q_kernel<<<(total_rows + 7) / 8, threads, (size_t)cols * sizeof(float)>>>(
      q_type, Wq, q_row_bytes, k_type, Wk, k_row_bytes, v_type, Wv,
      v_row_bytes, x, q, k, v, q_rows, kv_rows, cols);
}

// Fuses attention output GEMV and its residual write, avoiding the temporary
// output vector round trip and one add launch per layer.
__global__ void gemv_q_residual(int type, const uint8_t* W, const float* in,
                                float* x, int rows, int cols, int row_bytes) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = in[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  const float out = row_dot(type, W + (size_t)row * row_bytes, xs, cols, lane);
  if (lane == 0) x[row] += out;
}

inline void launch_gemv_q_residual(int type, const uint8_t* W, const float* in,
                                   float* x, int rows, int cols, int row_bytes) {
  constexpr int threads = 256;
  gemv_q_residual<<<(rows + 7) / 8, threads, (size_t)cols * sizeof(float)>>>(
      type, W, in, x, rows, cols, row_bytes);
}

// One fused launch for gate projection, up projection and SiLU multiplication.
// Every block produces eight rows and stages the input once.
__global__ void expert_gate_up_silu_kernel(int type, const uint8_t* Wg,
                                           const uint8_t* Wu, const float* x,
                                           float* gate, int d_ff, int cols,
                                           int row_bytes) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= d_ff) return;
  const float g = row_dot(type, Wg + (size_t)row * row_bytes, xs, cols, lane);
  const float u = row_dot(type, Wu + (size_t)row * row_bytes, xs, cols, lane);
  if (lane == 0) gate[row] = (g / (1.0f + expf(-g))) * u;
}

// One fused launch for down projection plus weighted residual. Expert kernels
// are issued on the same stream, so ordinary ordered writes are race-free.
__global__ void expert_down_residual_kernel(int type, const uint8_t* Wd,
                                            const float* gate, float* x,
                                            float w, int d_model, int cols,
                                            int row_bytes) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = gate[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= d_model) return;
  const float acc = row_dot(type, Wd + (size_t)row * row_bytes, xs, cols, lane);
  if (lane == 0) x[row] += w * acc;
}

// Per-layer expert dispatch copied once from pinned host memory. The selected
// expert bundles are not contiguous in VRAM, so this tiny pointer table lets
// one grid process all top-k experts without changing model residency layout.
struct ExpertDispatch {
  const uint8_t* gate[8];
  const uint8_t* up[8];
  const uint8_t* down[8];
  float weight[8];
};

// The dispatch table is passed BY VALUE: kernel parameters live in constant
// memory (broadcast-cached), so no thread ever reads the pointer table from
// global memory. The earlier by-pointer version made every thread load the
// whole 224-byte struct from global, which is why grouped dispatch used to
// lose to 16 serial launches.
__global__ void expert_group_gate_up_silu_kernel(
    int type, const ExpertDispatch d, const float* x, float* gates,
    int n_experts, int d_ff, int cols, int row_bytes) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int expert = blockIdx.y;
  if (expert >= n_experts) return;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= d_ff) return;
  const float g = row_dot(type, d.gate[expert] + (size_t)row * row_bytes,
                          xs, cols, lane);
  const float u = row_dot(type, d.up[expert] + (size_t)row * row_bytes,
                          xs, cols, lane);
  if (lane == 0)
    gates[(size_t)expert * d_ff + row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void expert_group_down_kernel(
    int type, const ExpertDispatch d, const float* gates, float* outs,
    int n_experts, int d_model, int cols, int row_bytes) {
  extern __shared__ float gs[];
  const int expert = blockIdx.y;
  if (expert >= n_experts) return;
  const float* gate = gates + (size_t)expert * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) gs[i] = gate[i];
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= d_model) return;
  const float v = row_dot(type, d.down[expert] + (size_t)row * row_bytes,
                          gs, cols, lane);
  if (lane == 0) outs[(size_t)expert * d_model + row] = v;
}

__global__ void expert_group_residual_kernel(const ExpertDispatch d,
                                             const float* outs, float* x,
                                             int n_experts, int d_model) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d_model) return;
  // Keep selected-expert accumulation in the existing j=0..k-1 order.
  float v = x[i];
  for (int j = 0; j < n_experts; ++j) v += d.weight[j] * outs[(size_t)j * d_model + i];
  x[i] = v;
}

inline void launch_expert_group_gate_up_silu(
    int type, const ExpertDispatch& dispatch, const float* x, float* gates,
    int n_experts, int d_ff, int cols, int row_bytes) {
  constexpr int threads = 256;
  dim3 grid((d_ff + 7) / 8, n_experts);
  expert_group_gate_up_silu_kernel<<<grid, threads, (size_t)cols * sizeof(float)>>>(
      type, dispatch, x, gates, n_experts, d_ff, cols, row_bytes);
}

inline void launch_expert_group_down(int type, const ExpertDispatch& dispatch,
                                     const float* gates, float* outs,
                                     int n_experts, int d_model, int cols,
                                     int row_bytes) {
  constexpr int threads = 256;
  dim3 grid((d_model + 7) / 8, n_experts);
  expert_group_down_kernel<<<grid, threads, (size_t)cols * sizeof(float)>>>(
      type, dispatch, gates, outs, n_experts, d_model, cols, row_bytes);
}

inline void launch_expert_group_residual(const ExpertDispatch& dispatch,
                                         const float* outs, float* x,
                                         int n_experts, int d_model) {
  constexpr int threads = 256;
  expert_group_residual_kernel<<<(d_model + threads - 1) / threads, threads>>>(
      dispatch, outs, x, n_experts, d_model);
}

// ---- GPU router epilogue: softmax over all logits + top-k select +
// renormalize, so the host round-trip shrinks from 512 B of logits + a CPU
// softmax/sort to one 64 B struct copy. Selection is by raw logit with
// lowest-index tie-break (host partial_sort left tie order unspecified;
// float ties are measure-zero). Softmax accumulates in double like the host.
struct RouteTopK {
  int ids[8];
  float w[8];
};

__global__ void router_topk8_kernel(const float* logits, int n_experts, int k,
                                    RouteTopK* out) {
  __shared__ float sl[128];
  __shared__ float red[128];
  const int t = threadIdx.x;
  const float v = (t < n_experts) ? logits[t] : -1e30f;
  sl[t] = v;
  red[t] = v;
  __syncthreads();
  for (int s = 64; s > 0; s >>= 1) {
    if (t < s) red[t] = fmaxf(red[t], red[t + s]);
    __syncthreads();
  }
  const float maxl = red[0];
  __syncthreads();
  __shared__ double se[128];
  se[t] = (t < n_experts) ? exp((double)(sl[t] - maxl)) : 0.0;
  __syncthreads();
  if (t == 0) {
    double denom = 0.0;
    for (int i = 0; i < n_experts; ++i) denom += se[i];  // host-order sum
    double ssum = 0.0;
    for (int j = 0; j < k; ++j) {
      int bi = -1;
      float bv = -1e30f;
      for (int i = 0; i < n_experts; ++i)
        if (sl[i] > bv) { bv = sl[i]; bi = i; }
      sl[bi] = -2e30f;  // remove from later rounds
      out->ids[j] = bi;
      const double p = se[bi] / denom;
      out->w[j] = (float)p;
      ssum += p;
    }
    for (int j = 0; j < k; ++j) out->w[j] = (float)(out->w[j] / ssum);
  }
}

// ---- GPU argmax over the vocab logits (two-stage), replacing a 608 KB
// logits D2H + host scan per token with a 4-byte copy. Lowest index wins
// ties, matching the host loop's strict-> comparison.
__global__ void argmax_stage1_kernel(const float* v, int n, float* pv, int* pi) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  const int t = threadIdx.x;
  float best = -1e30f;
  int besti = 0;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) {
    const float x = v[i];
    if (x > best) { best = x; besti = i; }
  }
  bv[t] = best;
  bi[t] = besti;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) {
      if (bv[t + s] > bv[t] || (bv[t + s] == bv[t] && bi[t + s] < bi[t])) {
        bv[t] = bv[t + s];
        bi[t] = bi[t + s];
      }
    }
    __syncthreads();
  }
  if (t == 0) { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
}

__global__ void argmax_stage2_kernel(const float* pv, const int* pi, int nparts,
                                     int* out) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  const int t = threadIdx.x;
  bv[t] = (t < nparts) ? pv[t] : -1e30f;
  bi[t] = (t < nparts) ? pi[t] : 0x7fffffff;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) {
      if (bv[t + s] > bv[t] || (bv[t + s] == bv[t] && bi[t + s] < bi[t])) {
        bv[t] = bv[t + s];
        bi[t] = bi[t + s];
      }
    }
    __syncthreads();
  }
  if (t == 0) out[0] = bi[0];
}

inline void launch_gemv_q(int type, const uint8_t* W, const float* x, float* y,
                          int rows, int cols, int row_bytes) {
  constexpr int threads = 256;
  const int nblocks = (rows + 7) / 8;
  gemv_q<<<nblocks, threads, (size_t)cols * sizeof(float)>>>(
      type, W, x, y, rows, cols, row_bytes);
}

inline void launch_expert_gate_up_silu(int type, const uint8_t* Wg,
                                       const uint8_t* Wu, const float* x,
                                       float* gate, int d_ff, int cols,
                                       int row_bytes) {
  constexpr int threads = 256;
  const int nblocks = (d_ff + 7) / 8;
  expert_gate_up_silu_kernel<<<nblocks, threads, (size_t)cols * sizeof(float)>>>(
      type, Wg, Wu, x, gate, d_ff, cols, row_bytes);
}

inline void launch_expert_down_residual(int type, const uint8_t* Wd,
                                        const float* gate, float* x, float w,
                                        int d_model, int cols, int row_bytes) {
  constexpr int threads = 256;
  const int nblocks = (d_model + 7) / 8;
  expert_down_residual_kernel<<<nblocks, threads, (size_t)cols * sizeof(float)>>>(
      type, Wd, gate, x, w, d_model, cols, row_bytes);
}

}  // namespace moex
