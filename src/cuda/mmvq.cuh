// MoEx clean-room MMVQ-style path (algorithms studied from llama.cpp mmvq /
// quantize / vecdotq — no source copied).
//
// Keys vs the failed earlier attempt:
// 1) Quantize activations to Q8 ONCE into global buffers (not every GEMV block).
// 2) Stage only Q8 into smem (~3 KiB @ 2048), never float[cols]+Q8 together.
// 3) Q4_K/Q6_K dots use int8×nibble + __dp4a where scales allow.
#pragma once

#include "cuda/kernels.cuh"

namespace moex {

// SoA Q8 activation: d[nblk] scales + q[cols] int8 values (cols = 32*nblk).
__host__ __device__ inline size_t q8_smem_bytes(int cols) {
  return (size_t)(cols / 32) * sizeof(float) + (size_t)cols;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

// One block = one Q8 group of 32. Grid: nblk.
__global__ void quantize_q8_1_kernel(const float* x, float* d_out, int8_t* q_out,
                                     int nblk) {
  const int b = blockIdx.x;
  if (b >= nblk) return;
  const int lane = threadIdx.x;
  const float xi = x[b * 32 + lane];
  float amax = warp_reduce_max(fabsf(xi));
  const float d = amax / 127.0f;
  const float id = (amax > 0.0f) ? (1.0f / d) : 0.0f;
  int q = __float2int_rn(xi * id);
  q = q < -127 ? -127 : (q > 127 ? 127 : q);
  q_out[b * 32 + lane] = (int8_t)q;
  if (lane == 0) d_out[b] = d;
}

// Batch of vectors (e.g. 8 expert gate outs). grid: (nblk, nvec)
__global__ void quantize_q8_1_batch_kernel(const float* x, float* d_out,
                                           int8_t* q_out, int cols, int nvec) {
  const int vec = blockIdx.y;
  const int b = blockIdx.x;
  const int nblk = cols >> 5;
  if (vec >= nvec || b >= nblk) return;
  const int lane = threadIdx.x;
  const float* xv = x + (size_t)vec * cols;
  float* dv = d_out + (size_t)vec * nblk;
  int8_t* qv = q_out + (size_t)vec * cols;
  const float xi = xv[b * 32 + lane];
  float amax = warp_reduce_max(fabsf(xi));
  const float d = amax / 127.0f;
  const float id = (amax > 0.0f) ? (1.0f / d) : 0.0f;
  int q = __float2int_rn(xi * id);
  q = q < -127 ? -127 : (q > 127 ? 127 : q);
  qv[b * 32 + lane] = (int8_t)q;
  if (lane == 0) dv[b] = d;
}

inline void launch_quantize_q8_1(const float* x, float* d_out, int8_t* q_out,
                                 int cols) {
  const int nblk = cols >> 5;
  quantize_q8_1_kernel<<<nblk, 32, 0, moex_launch_stream()>>>(x, d_out, q_out, nblk);
}

inline void launch_quantize_q8_1_batch(const float* x, float* d_out, int8_t* q_out,
                                       int cols, int nvec) {
  const int nblk = cols >> 5;
  dim3 grid(nblk, nvec);
  quantize_q8_1_batch_kernel<<<grid, 32, 0, moex_launch_stream()>>>(x, d_out, q_out, cols, nvec);
}

// Reconstruct Q8 → float into a global scratch (keeps GEMV kernels unchanged).
__global__ void dequant_q8_1_kernel(const float* xd, const int8_t* xq, float* out,
                                    int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cols) out[i] = xd[i >> 5] * (float)xq[i];
}

inline void launch_dequant_q8_1(const float* xd, const int8_t* xq, float* out,
                                int cols) {
  dequant_q8_1_kernel<<<(cols + 255) / 256, 256, 0, moex_launch_stream()>>>(xd, xq, out, cols);
}

__device__ __forceinline__ void stage_q8_smem(char* raw, const float* xd_g,
                                               const int8_t* xq_g, int cols,
                                               float*& xd, int8_t*& xq) {
  xd = reinterpret_cast<float*>(raw);
  xq = reinterpret_cast<int8_t*>(xd + cols / 32);
  const int nblk = cols >> 5;
  for (int i = threadIdx.x; i < nblk; i += blockDim.x) xd[i] = xd_g[i];
  for (int i = threadIdx.x; i < cols; i += blockDim.x) xq[i] = xq_g[i];
  __syncthreads();
}

}  // namespace moex
