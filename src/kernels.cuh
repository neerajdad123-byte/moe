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

// ---- K3: router: softmax over E logits, top-K (ties -> lower index), raw probs (norm_topk_prob=false) ----
// One block of E threads (E <= 1024, multiple of 32).
__global__ void router_topk_k(const float* __restrict__ logits, int E, int K, int norm,
                              int* __restrict__ idx, float* __restrict__ w) {
    __shared__ float sv[32]; __shared__ int si[32];
    int e = threadIdx.x, lane = e & 31, wid = e >> 5, nw = blockDim.x >> 5;
    float l = logits[e];
    float m = l;  // block max
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) sv[wid] = m;
    __syncthreads();
    m = sv[0]; for (int i = 1; i < nw; i++) m = fmaxf(m, sv[i]);
    float p = __expf(l - m);
    p /= block_sum(p);
    float sel = p, total = 0.f;
    for (int k = 0; k < K; k++) {
        float bv = sel; int bi = e;
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __syncthreads();
        if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
        __syncthreads();
        bv = sv[0]; bi = si[0];
        for (int i = 1; i < nw; i++) if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) { bv = sv[i]; bi = si[i]; }
        if (e == bi) sel = -1.f;
        if (e == 0) { idx[k] = bi; w[k] = bv; }
        total += bv;
    }
    if (norm && e < K) { __syncthreads(); w[e] /= total; }
}

inline void router_topk(const float* logits, int E, int K, bool norm, int* idx, float* w, cudaStream_t st = 0) {
    router_topk_k<<<1, E, 0, st>>>(logits, E, K, norm, idx, w);
}

// ---- K4: routed experts, int4 gs64 symmetric ----
// Slot layout (= container): [gate IxH | up IxH | down HxI] packed int4 (low nibble first),
// then f32 scales [gate Ix(H/64) | up Ix(H/64) | down Hx(I/64)].
constexpr int EH = 2048, EI = 512, EGS = 64;
constexpr size_t EXPERT_Q_BYTES = (size_t)3 * EH * EI / 2;                           // 1,572,864
constexpr size_t EXPERT_BYTES = EXPERT_Q_BYTES + (size_t)3 * EI * (EH / EGS) * 4;   // + 196,608

// 16 bytes = 32 int4 values; dot with 32 consecutive fp32 x values.
__device__ __forceinline__ float dot32_i4(uint4 u, const float* __restrict__ x) {
    const uint32_t q[4] = {u.x, u.y, u.z, u.w};
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int v = (int)((q[j] >> (4 * n)) & 0xF);
            acc += (float)(v - ((v & 8) << 1)) * x[8 * j + n];
        }
    }
    return acc;
}

// act[k][I] = silu(gate_k . x) * (up_k . x). One warp per (expert, row); rows are 1024 B = 64 uint4.
template <int WARPS>
__global__ void expert_gate_up_k(const uint8_t* const* __restrict__ slots, const float* __restrict__ x,
                                 float* __restrict__ act) {
    __shared__ __align__(16) float xs[EH];
    for (int i = threadIdx.x; i < EH; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    int r = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int k = r / EI, row = r % EI;
    const uint8_t* s = slots[k];
    const uint4* g = reinterpret_cast<const uint4*>(s + (size_t)row * (EH / 2));
    const uint4* u = reinterpret_cast<const uint4*>(s + (size_t)(EI + row) * (EH / 2));
    const float* sc = reinterpret_cast<const float*>(s + EXPERT_Q_BYTES);
    const float* gsc = sc + row * (EH / EGS);
    const float* usc = sc + EI * (EH / EGS) + row * (EH / EGS);
    float ag = 0.f, au = 0.f;
#pragma unroll
    for (int c = lane; c < EH / 32; c += 32) {  // uint4 c covers x[32c..32c+31], group c/2
        uint4 qg = __ldcs(g + c), qu = __ldcs(u + c);
        ag += dot32_i4(qg, xs + 32 * c) * gsc[c >> 1];
        au += dot32_i4(qu, xs + 32 * c) * usc[c >> 1];
    }
    ag = warp_sum(ag); au = warp_sum(au);
    if (lane == 0) act[r] = ag / (1.f + __expf(-ag)) * au;
}

// out[H] = sum_k w[k] * (down_k . act_k). One warp per output row; a down row is 256 B = 16 uint4,
// so a warp covers two experts per iteration.
template <int WARPS>
__global__ void expert_down_k(const uint8_t* const* __restrict__ slots, const float* __restrict__ w, int K,
                              const float* __restrict__ act, float* __restrict__ out) {
    extern __shared__ __align__(16) float as[];  // [K][EI]
    for (int i = threadIdx.x; i < K * EI; i += blockDim.x) as[i] = act[i];
    __syncthreads();
    int d = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int half_ = lane >> 4, c = lane & 15;
    float acc = 0.f;
    for (int k = half_; k < K; k += 2) {
        const uint8_t* s = slots[k];
        uint4 q = __ldcs(reinterpret_cast<const uint4*>(s + (size_t)2 * EI * (EH / 2) + (size_t)d * (EI / 2)) + c);
        const float* sc = reinterpret_cast<const float*>(s + EXPERT_Q_BYTES) + 2 * EI * (EH / EGS) + d * (EI / EGS);
        acc += w[k] * sc[c >> 1] * dot32_i4(q, as + k * EI + 32 * c);
    }
    acc = warp_sum(acc);
    if (lane == 0) out[d] = acc;
}

// slots: device array of K slot pointers; w: device top-K weights; act: scratch [K*EI].
inline void experts_forward(const uint8_t* const* slots, const float* w, int K, const float* x,
                            float* act, float* out, cudaStream_t st = 0) {
    constexpr int WARPS = 8;
    expert_gate_up_k<WARPS><<<K * EI / WARPS, WARPS * 32, 0, st>>>(slots, x, act);
    expert_down_k<WARPS><<<EH / WARPS, WARPS * 32, K * EI * sizeof(float), st>>>(slots, w, K, act, out);
}

// ---- K5 helpers: shared expert ----
// a[i] = silu(a[i]) * b[i]
__global__ void silu_mul_k(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = a[i]; a[i] = g / (1.f + __expf(-g)) * b[i]; }
}
// out[d] += sigmoid(gate_logit[0]) * sh[d]
__global__ void gated_add_k(float* __restrict__ out, const float* __restrict__ sh,
                            const float* __restrict__ gate_logit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += sh[i] / (1.f + __expf(-gate_logit[0]));
}

// Shared expert added into `out`. Weights F16: gate/up [SI,H], down [H,SI], sgate [1,H].
// scratch: >= 2*SI + H + 1 floats.
inline void shared_expert_add(const half* Wg, const half* Wu, const half* Wd, const half* Wsg,
                              const float* x, float* out, float* scratch, int H, int SI, cudaStream_t st = 0) {
    float *g = scratch, *u = g + SI, *sh = u + SI, *sl = sh + H;
    gemv_f16(Wg, x, g, H, SI, st);
    gemv_f16(Wu, x, u, H, SI, st);
    gemv_f16(Wsg, x, sl, H, 1, st);
    silu_mul_k<<<(SI + 255) / 256, 256, 0, st>>>(g, u, SI);
    gemv_f16(Wd, g, sh, SI, H, st);
    gated_add_k<<<(H + 255) / 256, 256, 0, st>>>(out, sh, sl, H);
}
