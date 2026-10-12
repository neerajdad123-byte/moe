// Decode kernels. Activations fp32, weights as stored (F16 dense, int4 gs64 experts).
#pragma once
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <algorithm>

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

// ---- K10: int8 dense GEMV, y[r] = s[r] * (Q[r,:] . x), per-row symmetric scale ----
// R rows per warp (their loads are issued together for memory-level parallelism), 16-byte loads
// (16 int8) per lane. Requires I % 512 == 0.
template <int WARPS, int R>
__global__ void gemv_i8_k(const int8_t* __restrict__ Q, const float* __restrict__ s, const float* __restrict__ x,
                          float* __restrict__ y, int I, int O) {
    int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (row0 >= O) return;
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc[R] = {};
#pragma unroll 2
    for (int c = lane; c < I / 16; c += 32) {
        uint4 u[R];
#pragma unroll
        for (int r = 0; r < R; r++) u[r] = row0 + r < O ? __ldcs(reinterpret_cast<const uint4*>(Q + (size_t)(row0 + r) * I) + c) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float4 xx = __ldg(xv + 4 * c + j);
#pragma unroll
            for (int r = 0; r < R; r++) {
                uint32_t w = (&u[r].x)[j];
                acc[r] += (float)(int8_t)(w) * xx.x + (float)(int8_t)(w >> 8) * xx.y +
                          (float)(int8_t)(w >> 16) * xx.z + (float)(int8_t)(w >> 24) * xx.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        float v = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < O) y[row0 + r] = v * s[row0 + r];
    }
}

template <int WARPS, int R>
inline void gemv_i8(const int8_t* q, const float* s, const float* x, float* y, int I, int O, cudaStream_t st = 0) {
    int rows_per_block = WARPS * R;
    gemv_i8_k<WARPS, R><<<(O + rows_per_block - 1) / rows_per_block, WARPS * 32, 0, st>>>(q, s, x, y, I, O);
}

// F16 [O,I] -> int8 [O,I] + f32 scale per row (absmax / 127). One block per row.
__global__ void quantize_rows_i8_k(const half* __restrict__ W, int8_t* __restrict__ Q, float* __restrict__ s, int I) {
    const half* wr = W + (size_t)blockIdx.x * I;
    float m = 0.f;
    for (int i = threadIdx.x; i < I; i += blockDim.x) m = fmaxf(m, fabsf(__half2float(wr[i])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = 0.f; for (int i = 0; i < (int)blockDim.x / 32; i++) m = fmaxf(m, red[i]);
    float sc = m > 0.f ? m / 127.f : 1.f;
    for (int i = threadIdx.x; i < I; i += blockDim.x)
        Q[(size_t)blockIdx.x * I + i] = (int8_t)__float2int_rn(__half2float(wr[i]) / sc);
    if (threadIdx.x == 0) s[blockIdx.x] = sc;
}

// A dense matrix in whichever form it was loaded: F16 (exact), int8 + row scales, or int4 gs64 + group scales
// (the experts' format: low nibble first, s is [O][I/64]).
struct Mat {
    const half* f16 = nullptr;
    const int8_t* q = nullptr;
    const float* s = nullptr;
    const uint8_t* q4 = nullptr;
};

template <int WARPS>
__global__ void gemv_q4_k(const uint8_t* __restrict__ Q, const float* __restrict__ s, const float* __restrict__ x,
                          float* __restrict__ y, int I, int O);
inline int persistent_blocks();

inline void gemv(const Mat& m, const float* x, float* y, int I, int O, cudaStream_t st = 0) {
    constexpr int WARPS = 8;
    if (m.q4) gemv_q4_k<WARPS><<<std::min(persistent_blocks(), (O + WARPS - 1) / WARPS), WARPS * 32, (I + I / 8) * 4, st>>>(m.q4, m.s, x, y, I, O);
    else if (m.q) gemv_i8<WARPS, 1>(m.q, m.s, x, y, I, O, st);
    else gemv_f16(m.f16, x, y, I, O, st);
}

// ---- K12: prefill GEMM, Y[S][O] = X[S][I] . W[O][I]^T (* row scale), on tensor cores ----
// Weights are int8 (exact in fp16: they are integers; the row scale is applied after the fp32 accumulation)
// or F16. Activations are rounded to fp16. Block tile 64 rows x 64 tokens x 32 K, 4 warps of 32x32.
// Requires I % 32 == 0.
template <typename WT, bool Q4 = false>
__global__ void gemm_wmma_k(const WT* __restrict__ W, const float* __restrict__ s, const float* __restrict__ X,
                            float* __restrict__ Y, int S, int I, int O) {
    using namespace nvcuda;
    constexpr int BM = 64, BN = 64, BK = 32, LD = BK + 8;  // +8 halves: no bank conflicts on fragment loads
    __shared__ __align__(32) half As[BM * LD], Bs[BN * LD];
    __shared__ __align__(32) float Cs[4][32 * 32];
    const int o0 = blockIdx.x * BM, s0 = blockIdx.y * BN, warp = threadIdx.x >> 5, wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(c[i][j], 0.f);
    for (int k0 = 0; k0 < I; k0 += BK) {
        for (int e = threadIdx.x; e < BM * BK; e += blockDim.x) {
            int r = e / BK, k = e % BK, o = o0 + r, t = s0 + r;
            float wv = 0.f;
            if (o < O) {
                if constexpr (Q4) {  // int4 gs64: the group scale goes into the tile (no row scale afterwards)
                    size_t i = (size_t)o * I + k0 + k;
                    int b = reinterpret_cast<const uint8_t*>(W)[i >> 1], v = (i & 1) ? b >> 4 : b & 0xF;
                    wv = (float)(v - ((v & 8) << 1)) * s[(size_t)o * (I / 64) + ((k0 + k) >> 6)];
                } else wv = (float)W[(size_t)o * I + k0 + k];
            }
            As[r * LD + k] = __float2half(wv);
            Bs[r * LD + k] = t < S ? __float2half(X[(size_t)t * I + k0 + k]) : __float2half(0.f);
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], As + (wm + 16 * i) * LD + kk, LD);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], Bs + (wn + 16 * j) * LD + kk, LD);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
    float* cs = Cs[warp];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
        wmma::store_matrix_sync(cs + (16 * i) * 32 + 16 * j, c[i][j], 32, wmma::mem_row_major);
    __syncwarp();
    for (int e = threadIdx.x & 31; e < 32 * 32; e += 32) {
        int r = e / 32, n = e % 32, o = o0 + wm + r, t = s0 + wn + n;
        if (o < O && t < S) Y[(size_t)t * O + o] = cs[e] * (s && !Q4 ? s[o] : 1.f);
    }
}

// Y[S][O] = X[S][I] . W^T for a dense matrix in either form.
inline void gemm(const Mat& m, const float* X, float* Y, int S, int I, int O, cudaStream_t st = 0) {
    dim3 grid((O + 63) / 64, (S + 63) / 64);
    if (m.q4) gemm_wmma_k<uint8_t, true><<<grid, 128, 0, st>>>(m.q4, m.s, X, Y, S, I, O);
    else if (m.q) gemm_wmma_k<int8_t><<<grid, 128, 0, st>>>(m.q, m.s, X, Y, S, I, O);
    else gemm_wmma_k<half><<<grid, 128, 0, st>>>(m.f16, nullptr, X, Y, S, I, O);
}

// ---- K3: router: softmax over E logits, top-K (ties -> lower index), raw probs (norm_topk_prob=false) ----
// One block of E threads (E <= 1024, multiple of 32).
__global__ void router_topk_k(const float* __restrict__ logits, int E, int K, int norm,
                              int* __restrict__ idx, float* __restrict__ w) {
    __shared__ float sv[32]; __shared__ int si[32];
    logits += (size_t)blockIdx.x * E; idx += blockIdx.x * K; w += blockIdx.x * K;  // token blockIdx.x of a chunk
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

// T tokens: logits [T][E] -> idx, w [T][K].
inline void router_topk(const float* logits, int E, int K, bool norm, int* idx, float* w, cudaStream_t st = 0, int T = 1) {
    router_topk_k<<<T, E, 0, st>>>(logits, E, K, norm, idx, w);
}

// ---- K4: routed experts, int4 gs64 symmetric ----
// Slot layout (= container): [gate IxH | up IxH | down HxI] packed int4 (low nibble first),
// then f32 scales [gate Ix(H/64) | up Ix(H/64) | down Hx(I/64)].
constexpr int EH = 2048, EI = 512, EGS = 64, EGS_DENSE = 64;
constexpr size_t EXPERT_Q_BYTES = (size_t)3 * EH * EI / 2;                           // 1,572,864
constexpr size_t EXPERT_BYTES = EXPERT_Q_BYTES + (size_t)3 * EI * (EH / EGS) * 4;   // + 196,608

// 16 bytes = 32 int4 values; dot with 32 consecutive fp32 x values.
// Dequant without int->float converts (low-throughput on Ada): u = nibble ^ 8 is v + 8 in [0,15], and the
// float with bits 0x4B000000 | u is exactly 2^23 + u, so subtracting 2^23 + 8 gives v exactly.
// x must be 16-byte aligned (read as 8 float4).
__device__ __forceinline__ float dot32_i4(uint4 u, const float* __restrict__ x) {
    const uint32_t q[4] = {u.x ^ 0x88888888u, u.y ^ 0x88888888u, u.z ^ 0x88888888u, u.w ^ 0x88888888u};
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float4 a = xv[2 * j], b = xv[2 * j + 1];
        const float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int n = 0; n < 8; n++)
            acc += (__int_as_float(0x4B000000u | ((q[j] >> (4 * n)) & 0xFu)) - 8388616.f) * xs[n];
    }
    return acc;
}

// Shared-memory layout for activations read 32 at a time per lane: lane c starts at element 32c, a 128-byte
// stride that puts every lane on the same banks. Padding 4 floats per 32 shifts each lane by one 16-byte bank group.
__device__ __forceinline__ int xpad(int i) { return i + (i >> 5) * 4; }
__host__ __device__ constexpr int xpad_size(int n) { return n + n / 8; }

// ---- K11: int4 gs64 dense GEMV. x in padded shared memory (loaded once per block of a persistent grid),
// one warp per row at a time; a 16-byte load is 32 weights = half a scale group. Requires I % 64 == 0.
template <int WARPS>
__global__ void gemv_q4_k(const uint8_t* __restrict__ Q, const float* __restrict__ s, const float* __restrict__ x,
                          float* __restrict__ y, int I, int O) {
    extern __shared__ __align__(16) float xs[];  // [xpad_size(I)]
    for (int i = threadIdx.x; i < I; i += blockDim.x) xs[xpad(i)] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, G = I / EGS_DENSE;
    for (int row = blockIdx.x * WARPS + (threadIdx.x >> 5); row < O; row += gridDim.x * WARPS) {
        const uint4* wr = reinterpret_cast<const uint4*>(Q + (size_t)row * (I / 2));
        const float* sr = s + (size_t)row * G;
        float acc = 0.f;
#pragma unroll 2
        for (int c = lane; c < I / 32; c += 32) acc += dot32_i4(__ldcs(wr + c), xs + 36 * c) * sr[c >> 1];
        acc = warp_sum(acc);
        if (lane == 0) y[row] = acc;
    }
}

// F16 [O,I] -> int4 gs64 symmetric (scale = group absmax / 7, values clamped to [-8,7]). One block per row,
// one thread per 64-element group.
__global__ void quantize_rows_q4_k(const half* __restrict__ W, uint8_t* __restrict__ Q, float* __restrict__ s, int I) {
    const half* wr = W + (size_t)blockIdx.x * I;
    for (int g = threadIdx.x; g < I / 64; g += blockDim.x) {
        float m = 0.f;
        for (int i = 0; i < 64; i++) m = fmaxf(m, fabsf(__half2float(wr[g * 64 + i])));
        float sc = m > 0.f ? m / 7.f : 1.f;
        uint8_t* qr = Q + (size_t)blockIdx.x * (I / 2) + g * 32;
        for (int i = 0; i < 64; i += 2) {
            int a = max(-8, min(7, __float2int_rn(__half2float(wr[g * 64 + i]) / sc)));
            int b = max(-8, min(7, __float2int_rn(__half2float(wr[g * 64 + i + 1]) / sc)));
            qr[i / 2] = (uint8_t)((a & 0xF) | ((b & 0xF) << 4));
        }
        s[(size_t)blockIdx.x * (I / 64) + g] = sc;
    }
}

// Persistent grid: enough blocks to fill the GPU once, each loading its shared inputs once and then
// looping over rows (a block per 8 rows re-read the 8-16 KB of inputs ~500 times per layer).
inline int persistent_blocks() {
    static int n = 0;
    if (!n) { int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0); n = sms * 4; }
    return n;
}

// act[k][I] = silu(gate_k . x) * (up_k . x). One warp per (expert, row) at a time; rows are 1024 B = 64 uint4.
// nh: optional device count of valid experts (<= K).
template <int WARPS>
__global__ void expert_gate_up_k(const uint8_t* const* __restrict__ slots, int K, const int* __restrict__ nh,
                                 const float* __restrict__ x, float* __restrict__ act) {
    __shared__ __align__(16) float xs[xpad_size(EH)];
    for (int i = threadIdx.x; i < EH; i += blockDim.x) xs[xpad(i)] = x[i];
    __syncthreads();
    const int rows = (nh ? *nh : K) * EI, lane = threadIdx.x & 31;
    for (int r = blockIdx.x * WARPS + (threadIdx.x >> 5); r < rows; r += gridDim.x * WARPS) {
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
            ag += dot32_i4(qg, xs + 36 * c) * gsc[c >> 1];
            au += dot32_i4(qu, xs + 36 * c) * usc[c >> 1];
        }
        ag = warp_sum(ag); au = warp_sum(au);
        if (lane == 0) act[r] = ag / (1.f + __expf(-ag)) * au;
    }
}

// out[H] = sum_k w[k] * (down_k . act_k). One warp per output row at a time; a down row is 256 B = 16 uint4,
// so a warp covers two experts per iteration.
template <int WARPS>
__global__ void expert_down_k(const uint8_t* const* __restrict__ slots, const float* __restrict__ w, int K,
                              const int* __restrict__ nh, const float* __restrict__ act, float* __restrict__ out) {
    if (nh) K = *nh;
    extern __shared__ __align__(16) float as[];  // [K][xpad_size(EI)]
    for (int i = threadIdx.x; i < K * EI; i += blockDim.x) as[(i / EI) * xpad_size(EI) + xpad(i % EI)] = act[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, half_ = lane >> 4, c = lane & 15;
    constexpr int KMAX = 8;  // = top-k; all of a row's loads are issued before any math (latency-bound otherwise)
    for (int d = blockIdx.x * WARPS + (threadIdx.x >> 5); d < EH; d += gridDim.x * WARPS) {
        uint4 q[KMAX / 2]; float sw[KMAX / 2];
#pragma unroll
        for (int i = 0; i < KMAX / 2; i++) {
            int k = 2 * i + half_;
            if (k < K) {
                const uint8_t* s = slots[k];
                q[i] = __ldcs(reinterpret_cast<const uint4*>(s + (size_t)2 * EI * (EH / 2) + (size_t)d * (EI / 2)) + c);
                sw[i] = w[k] * reinterpret_cast<const float*>(s + EXPERT_Q_BYTES)[2 * EI * (EH / EGS) + d * (EI / EGS) + (c >> 1)];
            }
        }
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < KMAX / 2; i++) {
            int k = 2 * i + half_;
            if (k < K) acc += sw[i] * dot32_i4(q[i], as + k * xpad_size(EI) + 36 * c);
        }
        acc = warp_sum(acc);
        if (lane == 0) out[d] = acc;
    }
}

// slots: device array of up to K slot pointers; w: device weights; act: scratch [K*EI].
// nh: optional device count of valid entries (<= K), so the GPU can decide hits without the host;
// with nh == 0 the output is all zeros.
inline void experts_forward(const uint8_t* const* slots, const float* w, int K, const float* x,
                            float* act, float* out, cudaStream_t st = 0, const int* nh = nullptr) {
    constexpr int WARPS = 8;
    int nb = persistent_blocks();
    expert_gate_up_k<WARPS><<<nb, WARPS * 32, 0, st>>>(slots, K, nh, x, act);
    expert_down_k<WARPS><<<nb, WARPS * 32, K * xpad_size(EI) * sizeof(float), st>>>(slots, w, K, nh, act, out);
}

// Batched slot-table writes, passed by value so each lands with the value it had when enqueued.
struct TableUpd {
    static constexpr int N = 32;
    int n = 0;
    int key[N];
    const uint8_t* ptr[N];
};
__global__ void table_k(const uint8_t** table, TableUpd u) {
    if ((int)threadIdx.x < u.n) table[u.key[threadIdx.x]] = u.ptr[threadIdx.x];
}

// ---- Prefill MoE: (token, expert) pairs grouped by expert, so each resident expert's weights are read once
// for all of its tokens. Pair p = t*K + rank; the host builds the work list.
struct PairItem { const uint8_t* slot; int start, count; };  // pairs[start, start+count) all use this expert
struct Pair { int t, p; float w; };

// act[p][row] = silu(gate . x_t) * (up . x_t) for every pair of item blockIdx.y. One warp per row; the row's
// weights stay in registers while the warp walks the item's tokens.
template <int WARPS>
__global__ void pairs_gate_up_k(const PairItem* __restrict__ items, const Pair* __restrict__ pairs,
                                const float* __restrict__ X, float* __restrict__ act) {
    const PairItem it = items[blockIdx.y];
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const uint4* g = reinterpret_cast<const uint4*>(it.slot + (size_t)row * (EH / 2));
    const uint4* u = reinterpret_cast<const uint4*>(it.slot + (size_t)(EI + row) * (EH / 2));
    const float* sc = reinterpret_cast<const float*>(it.slot + EXPERT_Q_BYTES);
    uint4 qg[2], qu[2]; float sg[2], su[2];
    for (int i = 0; i < 2; i++) {  // EH/32 = 64 uint4 per row: two per lane
        int c = lane + 32 * i;
        qg[i] = __ldcs(g + c); qu[i] = __ldcs(u + c);
        sg[i] = sc[row * (EH / EGS) + (c >> 1)]; su[i] = sc[EI * (EH / EGS) + row * (EH / EGS) + (c >> 1)];
    }
    for (int j = 0; j < it.count; j++) {
        const Pair pr = pairs[it.start + j];
        const float* x = X + (size_t)pr.t * EH;
        float ag = 0.f, au = 0.f;
        for (int i = 0; i < 2; i++) {
            int c = lane + 32 * i;
            ag += dot32_i4(qg[i], x + 32 * c) * sg[i];
            au += dot32_i4(qu[i], x + 32 * c) * su[i];
        }
        ag = warp_sum(ag); au = warp_sum(au);
        if (lane == 0) act[(size_t)pr.p * EI + row] = ag / (1.f + __expf(-ag)) * au;
    }
}

// ypair[p][d] = w_p * (down . act_p). One warp per output row; the row (256 B) is one uint4 per lane of a
// half-warp, and the two half-warps take alternate tokens.
template <int WARPS>
__global__ void pairs_down_k(const PairItem* __restrict__ items, const Pair* __restrict__ pairs,
                             const float* __restrict__ act, float* __restrict__ ypair) {
    const PairItem it = items[blockIdx.y];
    const int d = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31, half_ = lane >> 4, c = lane & 15;
    const uint4 q = __ldcs(reinterpret_cast<const uint4*>(it.slot + (size_t)2 * EI * (EH / 2) + (size_t)d * (EI / 2)) + c);
    const float s = reinterpret_cast<const float*>(it.slot + EXPERT_Q_BYTES)[2 * EI * (EH / EGS) + d * (EI / EGS) + (c >> 1)];
    for (int j0 = 0; j0 < it.count; j0 += 2) {
        int j = j0 + half_;
        float v = 0.f;
        Pair pr = pairs[it.start + (j < it.count ? j : j0)];
        if (j < it.count) v = s * dot32_i4(q, act + (size_t)pr.p * EI + 32 * c);
        for (int o = 8; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);  // sum within each half-warp
        if (c == 0 && j < it.count) ypair[(size_t)pr.p * EH + d] = pr.w * v;
    }
}

// moe[t][d] = sum over ranks of ypair[t*K + r][d] (GPU pairs; others are zero) + ycpu[t][d] (CPU pairs), in rank order.
__global__ void pairs_reduce_k(const float* __restrict__ ypair, const float* __restrict__ ycpu, float* __restrict__ moe, int K, int D) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int r = 0; r < K; r++) acc += ypair[((size_t)t * K + r) * D + d];
    moe[(size_t)t * D + d] = acc + ycpu[(size_t)t * D + d];
}

inline void pairs_forward(const PairItem* items, int n_items, const Pair* pairs, const float* X, float* act, float* ypair,
                          cudaStream_t st = 0) {
    constexpr int WARPS = 8;
    if (!n_items) return;
    pairs_gate_up_k<WARPS><<<dim3(EI / WARPS, n_items), WARPS * 32, 0, st>>>(items, pairs, X, act);
    pairs_down_k<WARPS><<<dim3(EH / WARPS, n_items), WARPS * 32, 0, st>>>(items, pairs, act, ypair);
}

// rows: out[t][i] = silu(gu[t][i]) * gu[t][SI + i]  (fused [gate; up] GEMM output -> compact [T][SI])
__global__ void silu_mul_rows_k(const float* __restrict__ gu, float* __restrict__ out, int SI) {
    int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SI) { float g = gu[(size_t)t * 2 * SI + i]; out[(size_t)t * SI + i] = g / (1.f + __expf(-g)) * gu[(size_t)t * 2 * SI + SI + i]; }
}
// rows: out[t][d] += sigmoid(gl[t]) * sh[t][d]
__global__ void gated_add_rows_k(float* __restrict__ out, const float* __restrict__ sh, const float* __restrict__ gl, int D) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < D) out[(size_t)t * D + d] += sh[(size_t)t * D + d] / (1.f + __expf(-gl[t]));
}

// Hits of this layer's routing, decided on the GPU from the device slot table (null = not resident).
// Writes the compacted slot pointers and weights, and their count. One warp.
__global__ void route_hits_k(const int* __restrict__ idx, const float* __restrict__ w, int K, int key_base,
                             const uint8_t* const* __restrict__ table, const uint8_t** __restrict__ slots,
                             float* __restrict__ hw, int* __restrict__ nh) {
    int j = threadIdx.x;
    const uint8_t* p = j < K ? table[key_base + idx[j]] : nullptr;
    unsigned m = __ballot_sync(0xffffffffu, p != nullptr);
    if (p) { int o = __popc(m & ((1u << j) - 1)); slots[o] = p; hw[o] = w[j]; }
    if (j == 0) *nh = __popc(m);
}

// ---- K5 helpers: shared expert ----
// a[i] = silu(a[i]) * b[i]
__global__ void silu_mul_k(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = a[i]; a[i] = g / (1.f + __expf(-g)) * b[i]; }
}
// Routing for the host, written straight into mapped host memory: MoE input h, top-K ids and weights.
__global__ void export_route_k(const float* __restrict__ h, const int* __restrict__ idx, const float* __restrict__ w,
                               float* h_out, int* idx_out, float* w_out, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) h_out[i] = h[i];
    if (i < K) { idx_out[i] = idx[i]; w_out[i] = w[i]; }
}

// out[i] = y[i]
__global__ void copy_k(float* __restrict__ out, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = y[i];
}
// out[i] += y[i]
__global__ void add_k(float* __restrict__ out, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += y[i];
}
// out[d] += sigmoid(gate_logit[0]) * sh[d]
__global__ void gated_add_k(float* __restrict__ out, const float* __restrict__ sh,
                            const float* __restrict__ gate_logit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += sh[i] / (1.f + __expf(-gate_logit[0]));
}

// ---- K6: per-head RMSNorm(1+w) + partial RoPE, then append k/v to the cache ----
// q: [H][2*HD] (query ++ gate per head, from q_proj) -> query written to qo [H][HD].
// k, v: [KV][HD]. Cache layout [KV][max_t][HD] fp32.
// shortcut: fp32 KV cache (exact vs reference); F16/int8 KV when long contexts need the VRAM.
constexpr int AHD = 256;
__global__ void qk_norm_rope_k(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                               const half* __restrict__ qn, const half* __restrict__ kn,
                               float* __restrict__ qo, float* __restrict__ Kc, float* __restrict__ Vc,
                               int H, int KV, const int* __restrict__ pos_d, int max_t, int rot, float theta, float eps, int ld) {
    __shared__ float t[AHD];
    // token blockIdx.y of a chunk is at *pos_d + y (device-side so a captured CUDA graph sees each token's position)
    const int tok = blockIdx.y, pos = *pos_d + tok;
    q += (size_t)tok * ld; k += (size_t)tok * ld; v += (size_t)tok * ld; qo += (size_t)tok * H * AHD;
    int hh = blockIdx.x, d = threadIdx.x;  // blocks [0,H) query heads, [H,H+KV) key heads
    bool isq = hh < H;
    float x = isq ? q[hh * 2 * AHD + d] : k[(hh - H) * AHD + d];
    float r = rsqrtf(block_sum(x * x) / AHD + eps);
    x = x * r * (1.f + __half2float((isq ? qn : kn)[d]));
    t[d] = x;
    __syncthreads();
    int h2 = rot / 2;
    if (d < rot) {
        int j = d < h2 ? d : d - h2;
        float ang = (float)pos * powf(theta, -2.f * j / rot);
        float cs = cosf(ang), sn = sinf(ang);
        x = d < h2 ? t[d] * cs - t[d + h2] * sn : t[d] * cs + t[d - h2] * sn;
    }
    if (isq) qo[hh * AHD + d] = x;
    else {
        size_t o = ((size_t)(hh - H) * max_t + pos) * AHD + d;
        Kc[o] = x;
        Vc[o] = v[(hh - H) * AHD + d];
    }
}

// ---- K7: decode attention, one block (256 threads = head dim) per query head ----
// ctx = softmax(q.K^T / sqrt(HD)) V over t in [0, T); out = ctx * sigmoid(gate).
// shortcut: whole score row in shared memory (T <= ~24k with opt-in smem); split-K when contexts grow.
__global__ void attn_decode_k(const float* __restrict__ qo, const float* __restrict__ q_gate,
                              const float* __restrict__ Kc, const float* __restrict__ Vc,
                              float* __restrict__ out, int H, int KV, const int* __restrict__ pos_d, int max_t, int ld) {
    extern __shared__ float sc[];  // [max_t]
    const int tok = blockIdx.y, T = *pos_d + tok + 1;  // causal: token tok sees positions up to its own
    qo += (size_t)tok * H * AHD; q_gate += (size_t)tok * ld; out += (size_t)tok * H * AHD;
    __shared__ __align__(16) float qs[AHD];
    int h = blockIdx.x, kvh = h / (H / KV), tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    qs[tid] = qo[h * AHD + tid];
    __syncthreads();
    const float* K = Kc + (size_t)kvh * max_t * AHD;
    const float* V = Vc + (size_t)kvh * max_t * AHD;
    const float scale = rsqrtf((float)AHD);
    for (int t = wid; t < T; t += blockDim.x / 32) {  // warp per key, 8 dims per lane
        const float4* kr = reinterpret_cast<const float4*>(K + (size_t)t * AHD) + lane * 2;
        const float4* qr = reinterpret_cast<const float4*>(qs) + lane * 2;
        float4 a = kr[0], b = kr[1], qa = qr[0], qb = qr[1];
        float s = warp_sum(a.x * qa.x + a.y * qa.y + a.z * qa.z + a.w * qa.w + b.x * qb.x + b.y * qb.y + b.z * qb.z + b.w * qb.w);
        if (lane == 0) sc[t] = s * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int t = tid; t < T; t += blockDim.x) m = fmaxf(m, sc[t]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float red[32];
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = red[0]; for (int i = 1; i < blockDim.x / 32; i++) m = fmaxf(m, red[i]);
    float s = 0.f;
    for (int t = tid; t < T; t += blockDim.x) { float e = __expf(sc[t] - m); sc[t] = e; s += e; }
    float inv = 1.f / block_sum(s);  // block_sum syncs, so sc[] is complete after it
    float acc = 0.f;
    for (int t = 0; t < T; t++) acc += sc[t] * V[(size_t)t * AHD + tid];
    float g = q_gate[h * 2 * AHD + AHD + tid];
    out[h * AHD + tid] = acc * inv / (1.f + __expf(-g));
}

// pos_d: device int, the position of the first token. Shared memory is sized for max_t (fixed per launch
// shape, so the same launch can be replayed from a graph at any position).
// T tokens (a prefill chunk): q/k/v rows have stride ld; all K/V are appended before any token attends.
// shortcut: one block per (head, token) re-reads K/V per token; a flash-style tile for long prompts.
inline void attn_decode(const float* q, const float* k, const float* v, const half* qn, const half* kn,
                        float* qo, float* Kc, float* Vc, float* out, int H, int KV, const int* pos_d, int max_t,
                        int rot, float theta, float eps, int T, int ld, cudaStream_t st = 0) {
    qk_norm_rope_k<<<dim3(H + KV, T), AHD, 0, st>>>(q, k, v, qn, kn, qo, Kc, Vc, H, KV, pos_d, max_t, rot, theta, eps, ld);
    attn_decode_k<<<dim3(H, T), AHD, max_t * sizeof(float), st>>>(qo, q, Kc, Vc, out, H, KV, pos_d, max_t, ld);
}

__global__ void set_int_k(int* p, int v) { *p = v; }

// ---- K8: Gated DeltaNet decode ----
// Conv: causal depthwise conv (kernel CK) over C channels with a carried ring [C][CK-1], then SiLU.
// T tokens in order (T = 1 for decode, a chunk for prefill). qkv rows have stride ld; out is [T][C].
__global__ void dn_conv_k(const float* __restrict__ qkv, int ld, const half* __restrict__ w, float* __restrict__ ring,
                          float* __restrict__ out, int C, int CKn, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* rg = ring + (size_t)c * (CKn - 1);
    const half* wc = w + (size_t)c * CKn;
    for (int t = 0; t < T; t++) {
        float x = qkv[(size_t)t * ld + c], acc = __half2float(wc[CKn - 1]) * x;
        for (int i = 0; i < CKn - 1; i++) acc += __half2float(wc[i]) * rg[i];
        for (int i = 0; i < CKn - 2; i++) rg[i] = rg[i + 1];
        rg[CKn - 2] = x;
        out[(size_t)t * C + c] = acc / (1.f + __expf(-acc));
    }
}

// Recurrence + gated RMSNorm over T tokens. One block per value head, thread = value column (DV threads);
// the state column S[:, v] lives in registers for the whole chunk: S is read once and written once per call.
// conv is [T][C]; z, a, b rows have stride ld; o is [T][VH*DNV].
constexpr int DNK = 128, DNV = 128;
__global__ void dn_recur_k(const float* __restrict__ conv, int C, const float* __restrict__ z,
                           const float* __restrict__ a, const float* __restrict__ b, int ld,
                           const half* __restrict__ A_log, const half* __restrict__ dt_bias,
                           const half* __restrict__ nw, float* __restrict__ S, float* __restrict__ o,
                           int VH, int KH, float eps, int T) {
    __shared__ float qs[DNK], ks[DNK];
    const int h = blockIdx.x, v = threadIdx.x, kh = h / (VH / KH);
    const float alog = expf(__half2float(A_log[h])), dtb = __half2float(dt_bias[h]), nwv = __half2float(nw[v]);
    float* Sh = S + (size_t)h * DNK * DNV;
    float col[DNK];
#pragma unroll
    for (int i = 0; i < DNK; i++) col[i] = Sh[i * DNV + v];
    for (int t = 0; t < T; t++) {
        const float* cv = conv + (size_t)t * C;
        float qv = cv[kh * DNK + v], kv_ = cv[KH * DNK + kh * DNK + v];  // DNK == DNV == blockDim
        float vin = cv[2 * KH * DNK + h * DNV + v];
        float qn = sqrtf(block_sum(qv * qv) + 1e-6f), kn = sqrtf(block_sum(kv_ * kv_) + 1e-6f);  // also orders qs/ks reuse
        qs[v] = qv / qn * rsqrtf((float)DNK);
        ks[v] = kv_ / kn;
        float beta = 1.f / (1.f + __expf(-b[(size_t)t * ld + h]));
        float ax = a[(size_t)t * ld + h] + dtb;
        float sp = ax > 20.f ? ax : log1pf(expf(ax));
        float eg = expf(-alog * sp);
        __syncthreads();
        float kvs = 0.f;
#pragma unroll
        for (int i = 0; i < DNK; i++) { col[i] *= eg; kvs += ks[i] * col[i]; }
        float delta = (vin - kvs) * beta, out = 0.f;
#pragma unroll
        for (int i = 0; i < DNK; i++) { col[i] += ks[i] * delta; out += qs[i] * col[i]; }
        float r = rsqrtf(block_sum(out * out) / DNV + eps);
        float zz = z[(size_t)t * ld + h * DNV + v];
        o[(size_t)t * VH * DNV + h * DNV + v] = out * r * nwv * (zz / (1.f + __expf(-zz)));
    }
#pragma unroll
    for (int i = 0; i < DNK; i++) Sh[i * DNV + v] = col[i];
}

// T tokens through conv + recurrence. qkv, z, a, b: rows of stride ld (one fused projection row per token).
inline void deltanet_core(const float* qkv, const float* z, const float* a, const float* b, int ld,
                          const half* conv_w, const half* A_log, const half* dt_bias, const half* nw,
                          float* ring, float* S, float* conv_scratch, float* o,
                          int C, int CKn, int VH, int KH, float eps, int T = 1, cudaStream_t st = 0) {
    dn_conv_k<<<(C + 255) / 256, 256, 0, st>>>(qkv, ld, conv_w, ring, conv_scratch, C, CKn, T);
    dn_recur_k<<<VH, DNV, 0, st>>>(conv_scratch, C, z, a, b, ld, A_log, dt_bias, nw, S, o, VH, KH, eps, T);
}

// ---- K9: argmax over n floats (ties -> lower index), two passes ----
// xi: candidate indices for x (second pass), or null for x's own positions.
__global__ void argmax_k(const float* __restrict__ x, const int* __restrict__ xi, int n, float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[32]; __shared__ int si[32];
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > bv) { bv = x[i]; bi = xi ? xi[i] : i; }
    for (int o = 16; o; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < blockDim.x / 32; i++) if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) { bv = sv[i]; bi = si[i]; }
        pv[blockIdx.x] = bv; pi[blockIdx.x] = bi;
    }
}
// sv: scratch of ARGMAX_BLOCKS+1 floats, si: ARGMAX_BLOCKS ints; the winning index lands in out[0]
constexpr int ARGMAX_BLOCKS = 160;
inline void argmax(const float* x, int n, float* sv, int* si, int* out, cudaStream_t st = 0) {
    argmax_k<<<ARGMAX_BLOCKS, 256, 0, st>>>(x, nullptr, n, sv, si);
    argmax_k<<<1, 256, 0, st>>>(sv, si, ARGMAX_BLOCKS, sv + ARGMAX_BLOCKS, out);
}

// Shared expert added into `out`. gate+up fused [2*SI,H], down [H,SI], sgate F16 [1,H].
// scratch: >= 2*SI + H + 1 floats.
inline void shared_expert_add(const Mat& Wgu, const Mat& Wd, const half* Wsg,
                              const float* x, float* out, float* scratch, int H, int SI, cudaStream_t st = 0) {
    float *g = scratch, *u = g + SI, *sh = u + SI, *sl = sh + H;
    gemv(Wgu, x, g, H, 2 * SI, st);  // fused [gate; up] -> g, u
    gemv_f16(Wsg, x, sl, H, 1, st);
    silu_mul_k<<<(SI + 255) / 256, 256, 0, st>>>(g, u, SI);
    gemv(Wd, g, sh, SI, H, st);
    gated_add_k<<<(H + 255) / 256, 256, 0, st>>>(out, sh, sl, H);
}
