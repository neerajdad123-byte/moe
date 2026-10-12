// Decode kernels. Activations fp32, weights as stored (F16 dense, int4 gs64 experts).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum; every thread gets the result. blockDim.x multiple of 32, <= 1024.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float s[32];
    v = warp_sum(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();  // s[] may still be read by a previous call
    if (l == 0) s[w] = v;
    __syncthreads();
    v = l < (blockDim.x >> 5) ? s[l] : 0.f;
    return warp_sum(v);
}

// ---- K1: RMSNorm, Qwen3.5 style: y = x * rsqrt(mean(x^2)+eps) * (1 + w) ----
// One block per row. If `delta` is set, x += delta first (residual add) and x is written back.
__global__ void rmsnorm_k(float* __restrict__ x, const float* __restrict__ delta,
                          const half* __restrict__ w, float* __restrict__ y, int D, float eps) {
    float* xr = x + (size_t)blockIdx.x * D;
    const float* dr = delta ? delta + (size_t)blockIdx.x * D : nullptr;
    float* yr = y + (size_t)blockIdx.x * D;
    float ss = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = xr[i];
        if (dr) { v += dr[i]; xr[i] = v; }
        ss += v * v;
    }
    float r = rsqrtf(block_sum(ss) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) yr[i] = xr[i] * r * (1.f + __half2float(w[i]));
}

inline void rmsnorm(float* x, const float* delta, const half* w, float* y, int D, float eps,
                    int rows = 1, cudaStream_t st = 0) {
    rmsnorm_k<<<rows, 256, 0, st>>>(x, delta, w, y, D, eps);
}

// ---- K2: dense GEMV, y[O] = W[O,I] (F16 row-major) . x[I] (fp32) ----
// One warp per row, 16-byte loads (8 halves) per lane. Requires I % 256 == 0.
template <int WARPS>
__global__ void gemv_f16_k(const half* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int I, int O) {
    int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= O) return;
    const uint4* wr = reinterpret_cast<const uint4*>(W + (size_t)row * I);
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
    for (int c = lane; c < I / 8; c += 32) {
        uint4 u = __ldcs(wr + c);  // weights are streamed once per token: don't pollute L2
        const half2* h = reinterpret_cast<const half2*>(&u);
        float4 a = __ldg(xv + 2 * c), b = __ldg(xv + 2 * c + 1);
        float2 f0 = __half22float2(h[0]), f1 = __half22float2(h[1]), f2 = __half22float2(h[2]), f3 = __half22float2(h[3]);
        acc += f0.x * a.x + f0.y * a.y + f1.x * a.z + f1.y * a.w + f2.x * b.x + f2.y * b.y + f3.x * b.z + f3.y * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

inline void gemv_f16(const half* W, const float* x, float* y, int I, int O, cudaStream_t st = 0) {
    constexpr int WARPS = 8;
    gemv_f16_k<WARPS><<<(O + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(W, x, y, I, O);
}
