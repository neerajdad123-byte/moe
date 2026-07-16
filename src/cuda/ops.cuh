// MoEx clean-room. Small reduction / selection kernels used every decode step.
// GPU argmax and GPU router top-k remove full-tensor D2H host syncs.
#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace moex {

// ---- Argmax over logits[n] → out_idx[0] ------------------------------------
// Multi-block reduction into a small scratch, then final pass.
// scratch: at least nblocks floats + nblocks ints (packed as float2-like pairs).
// Simpler path for vocab ~150k: one stage with atomic-free block reduce then
// a single-block final kernel.

struct ArgPair {
  float v;
  int i;
};

__device__ __forceinline__ ArgPair arg_max(ArgPair a, ArgPair b) {
  // Prefer higher value; on ties keep lower index (stable).
  if (b.v > a.v || (b.v == a.v && b.i < a.i)) return b;
  return a;
}

__global__ void argmax_partial(const float* logits, int n, float* part_v,
                               int* part_i) {
  extern __shared__ float sh[];
  float* sv = sh;
  int* si = (int*)(sh + blockDim.x);
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float v = -FLT_MAX;
  int idx = 0;
  // Grid-stride
  for (int j = i; j < n; j += gridDim.x * blockDim.x) {
    float x = logits[j];
    if (x > v || (x == v && j < idx)) {
      v = x;
      idx = j;
    }
  }
  sv[tid] = v;
  si[tid] = idx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sv[tid + s] > sv[tid] ||
          (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
        sv[tid] = sv[tid + s];
        si[tid] = si[tid + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    part_v[blockIdx.x] = sv[0];
    part_i[blockIdx.x] = si[0];
  }
}

__global__ void argmax_final(const float* part_v, const int* part_i, int npart,
                             int* out_idx) {
  extern __shared__ float sh[];
  float* sv = sh;
  int* si = (int*)(sh + blockDim.x);
  int tid = threadIdx.x;
  float v = -FLT_MAX;
  int idx = 0;
  for (int j = tid; j < npart; j += blockDim.x) {
    float x = part_v[j];
    int i = part_i[j];
    if (x > v || (x == v && i < idx)) {
      v = x;
      idx = i;
    }
  }
  sv[tid] = v;
  si[tid] = idx;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sv[tid + s] > sv[tid] ||
          (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
        sv[tid] = sv[tid + s];
        si[tid] = si[tid + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) out_idx[0] = si[0];
}

// Launch argmax. part_v/part_i must hold at least nblocks elements.
// nblocks chosen by caller (e.g. 128 or 256).
inline void launch_argmax(const float* logits, int n, float* part_v, int* part_i,
                          int nblocks, int* out_idx) {
  const int threads = 256;
  size_t smem = threads * (sizeof(float) + sizeof(int));
  argmax_partial<<<nblocks, threads, smem>>>(logits, n, part_v, part_i);
  // final: one block, up to 256 partials ideally; if nblocks > 256, stride
  int ft = 256;
  if (nblocks < ft) ft = nblocks;
  // pad to power of 2 friendly: use 256 threads always with stride in kernel
  argmax_final<<<1, 256, 256 * (sizeof(float) + sizeof(int))>>>(
      part_v, part_i, nblocks, out_idx);
}

// ---- Router top-k + softmax weights (n_experts <= 256, k <= 16) -------------
// Single block. Thread t owns expert t (if t < n).
// 1) block-reduce max logit
// 2) block-sum exp(logit - max) for full softmax denom
// 3) iterative top-k by logit (k passes)
// 4) gather p_i = exp(l_i-max)/denom for top-k, renorm into out_w
__global__ void router_topk_softmax(const float* logits, int n, int k,
                                    int* out_idx, float* out_w) {
  extern __shared__ float sh[];
  // layout: [n] logits copy, [n] scratch, [k] used flags via int after
  float* loc = sh;                  // n
  float* red = sh + n;              // n (reuse)
  int tid = threadIdx.x;

  float my = (tid < n) ? logits[tid] : -FLT_MAX;
  if (tid < n) loc[tid] = my;
  __syncthreads();

  // max
  float m = my;
  red[tid] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  m = red[0];
  __syncthreads();

  // sum exp
  float e = (tid < n) ? expf(loc[tid] - m) : 0.0f;
  red[tid] = e;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float denom = red[0];
  __syncthreads();

  // iterative top-k: mask used experts by setting loc to -inf after pick
  // work on a copy of logits in loc
  for (int t = 0; t < k; ++t) {
    // find max among remaining
    float bv = (tid < n) ? loc[tid] : -FLT_MAX;
    int bi = tid;
    red[tid] = bv;
    // also need index: pack via second array — reuse after barrier carefully
    __syncthreads();
    // pairwise max with index in two arrays
    // store idx in shared int region after 2*n floats
    int* ired = (int*)(sh + 2 * n);
    ired[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (red[tid + s] > red[tid] ||
            (red[tid + s] == red[tid] && ired[tid + s] < ired[tid])) {
          red[tid] = red[tid + s];
          ired[tid] = ired[tid + s];
        }
      }
      __syncthreads();
    }
    int win = ired[0];
    if (tid == 0) {
      out_idx[t] = win;
      // weight before renorm
      out_w[t] = expf(logits[win] - m) / denom;
    }
    __syncthreads();
    // mask winner
    if (tid == win) loc[tid] = -FLT_MAX;
    __syncthreads();
  }

  // renorm top-k weights
  if (tid == 0) {
    float s = 0.0f;
    for (int t = 0; t < k; ++t) s += out_w[t];
    float inv = 1.0f / s;
    for (int t = 0; t < k; ++t) out_w[t] *= inv;
  }
}

// n_experts must be <= 256, blockDim >= next_pow2(n) and >= n
inline void launch_router_topk(const float* logits, int n, int k, int* out_idx,
                               float* out_w) {
  int threads = 1;
  while (threads < n) threads <<= 1;
  if (threads < 32) threads = 32;
  // smem: 2*n floats + threads ints
  size_t smem =
      (size_t)(2 * n) * sizeof(float) + (size_t)threads * sizeof(int);
  router_topk_softmax<<<1, threads, smem>>>(logits, n, k, out_idx, out_w);
}

// Fused silu(gate)*up → gate (in-place on gate).
__global__ void silu_mul_kernel(float* gate, const float* up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

// x += scale * y
__global__ void axpy_kernel(float* x, const float* y, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += scale * y[i];
}

// x += y
__global__ void add_kernel(float* x, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += y[i];
}

}  // namespace moex
