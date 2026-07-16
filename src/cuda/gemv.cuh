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
__device__ __forceinline__ float row_dot(int type, const uint8_t* row,
                                          const float* x, int cols, int lane) {
  float acc = 0.0f;
  if (type == GT_Q4_K) {
    const int nsb = cols / 256;
    for (int sb = lane; sb < nsb; sb += 32)
      acc += dot_superblock_q4k(row + (size_t)sb * BQ4_K, x + sb * 256);
  } else if (type == GT_Q5_K) {
    const int nsb = cols / 256;
    for (int sb = lane; sb < nsb; sb += 32)
      acc += dot_superblock_q5k(row + (size_t)sb * BQ5_K, x + sb * 256);
  } else if (type == GT_Q6_K) {
    const int nsb = cols / 256;
    for (int sb = lane; sb < nsb; sb += 32)
      acc += dot_superblock_q6k(row + (size_t)sb * BQ6_K, x + sb * 256);
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

__global__ void expert_group_gate_up_silu_kernel(
    int type, const ExpertDispatch* dispatch, const float* x, float* gates,
    int n_experts, int d_ff, int cols, int row_bytes) {
  extern __shared__ float xs[];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  const int expert = blockIdx.y;
  if (expert >= n_experts) return;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= d_ff) return;
  const ExpertDispatch d = *dispatch;
  const float g = row_dot(type, d.gate[expert] + (size_t)row * row_bytes,
                          xs, cols, lane);
  const float u = row_dot(type, d.up[expert] + (size_t)row * row_bytes,
                          xs, cols, lane);
  if (lane == 0)
    gates[(size_t)expert * d_ff + row] = (g / (1.0f + expf(-g))) * u;
}

__global__ void expert_group_down_kernel(
    int type, const ExpertDispatch* dispatch, const float* gates, float* outs,
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
  const ExpertDispatch d = *dispatch;
  const float v = row_dot(type, d.down[expert] + (size_t)row * row_bytes,
                          gs, cols, lane);
  if (lane == 0) outs[(size_t)expert * d_model + row] = v;
}

__global__ void expert_group_residual_kernel(const ExpertDispatch* dispatch,
                                             const float* outs, float* x,
                                             int n_experts, int d_model) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d_model) return;
  const ExpertDispatch d = *dispatch;
  // Keep selected-expert accumulation in the existing j=0..k-1 order.
  float v = x[i];
  for (int j = 0; j < n_experts; ++j) v += d.weight[j] * outs[(size_t)j * d_model + i];
  x[i] = v;
}

inline void launch_expert_group_gate_up_silu(
    int type, const ExpertDispatch* dispatch, const float* x, float* gates,
    int n_experts, int d_ff, int cols, int row_bytes) {
  constexpr int threads = 256;
  dim3 grid((d_ff + 7) / 8, n_experts);
  expert_group_gate_up_silu_kernel<<<grid, threads, (size_t)cols * sizeof(float)>>>(
      type, dispatch, x, gates, n_experts, d_ff, cols, row_bytes);
}

inline void launch_expert_group_down(int type, const ExpertDispatch* dispatch,
                                     const float* gates, float* outs,
                                     int n_experts, int d_model, int cols,
                                     int row_bytes) {
  constexpr int threads = 256;
  dim3 grid((d_model + 7) / 8, n_experts);
  expert_group_down_kernel<<<grid, threads, (size_t)cols * sizeof(float)>>>(
      type, dispatch, gates, outs, n_experts, d_model, cols, row_bytes);
}

inline void launch_expert_group_residual(const ExpertDispatch* dispatch,
                                         const float* outs, float* x,
                                         int n_experts, int d_model) {
  constexpr int threads = 256;
  expert_group_residual_kernel<<<(d_model + threads - 1) / threads, threads>>>(
      dispatch, outs, x, n_experts, d_model);
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
