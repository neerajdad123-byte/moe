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

// Block-wide max; every thread gets the result. blockDim.x multiple of 32, <= 1024.
__device__ __forceinline__ float block_max(float v) {
    __shared__ float s[32];
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) s[w] = v;
    __syncthreads();
    v = l < (blockDim.x >> 5) ? s[l] : -INFINITY;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Set by wait_add_k when the GPU gave up waiting for the host in the whole-step decode graph: the kernels that change
// state the per-layer fallback needs (residual, KV cache, DeltaNet conv ring + state, MoE output, routing signal) then
// return at once for the rest of that graph launch. The host clears it before every step graph launch and fallback.
__device__ int g_skip;

// ---- K1: RMSNorm, Qwen3.5 style: y = x * rsqrt(mean(x^2)+eps) * (1 + w) ----
// One block per row. If `delta` is set, x += delta first (residual add) and x is written back.
__global__ void rmsnorm_k(float* __restrict__ x, const float* __restrict__ delta,
                          const half* __restrict__ w, float* __restrict__ y, int D, float eps) {
    if (g_skip) return;
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
// One warp per row, 16-byte loads (8 halves) per lane. Requires I % 256 == 0. blockIdx.y: one of `rows` packed x / y.
template <int WARPS>
__global__ void gemv_f16_k(const half* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int I, int O) {
    int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= O) return;
    x += (size_t)blockIdx.y * I; y += (size_t)blockIdx.y * O;
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

inline void gemv_f16(const half* W, const float* x, float* y, int I, int O, cudaStream_t st = 0, int rows = 1) {
    constexpr int WARPS = 8;
    gemv_f16_k<WARPS><<<dim3((O + WARPS - 1) / WARPS, rows), WARPS * 32, 0, st>>>(W, x, y, I, O);
}

// ---- K10: int8 dense GEMV, y[r] = s[r] * (Q[r,:] . x), per-row symmetric scale ----
// R rows per warp (their loads are issued together for memory-level parallelism), 16-byte loads
// (16 int8) per lane. Requires I % 512 == 0. NT vectors at once (x [NT][I], y [NT][O]): the weights are read once
// for all of them (the MTP round's 2 tokens).
template <int WARPS, int R, int NT = 1>
__global__ void gemv_i8_k(const int8_t* __restrict__ Q, const float* __restrict__ s, const float* __restrict__ x,
                          float* __restrict__ y, int I, int O) {
    int row0 = (blockIdx.x * WARPS + (threadIdx.x >> 5)) * R, lane = threadIdx.x & 31;
    if (row0 >= O) return;
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc[NT][R] = {};
#pragma unroll 2
    for (int c = lane; c < I / 16; c += 32) {
        uint4 u[R];
#pragma unroll
        for (int r = 0; r < R; r++) u[r] = row0 + r < O ? __ldcs(reinterpret_cast<const uint4*>(Q + (size_t)(row0 + r) * I) + c) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int t = 0; t < NT; t++) {
                float4 xx = __ldg(xv + (size_t)t * (I / 4) + 4 * c + j);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    uint32_t w = (&u[r].x)[j];
                    acc[t][r] += (float)(int8_t)(w) * xx.x + (float)(int8_t)(w >> 8) * xx.y +
                                 (float)(int8_t)(w >> 16) * xx.z + (float)(int8_t)(w >> 24) * xx.w;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
        for (int r = 0; r < R; r++) {
            float v = warp_sum(acc[t][r]);
            if (lane == 0 && row0 + r < O) y[(size_t)t * O + row0 + r] = v * s[row0 + r];
        }
}

template <int WARPS, int R, int NT = 1>
inline void gemv_i8(const int8_t* q, const float* s, const float* x, float* y, int I, int O, cudaStream_t st = 0) {
    int rows_per_block = WARPS * R;
    gemv_i8_k<WARPS, R, NT><<<(O + rows_per_block - 1) / rows_per_block, WARPS * 32, 0, st>>>(q, s, x, y, I, O);
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

// Research (MOEX_FQ): quantize F16 rows to `bits` per `gs`-group and write the dequantized values back, so a format's
// quality can be measured end to end before it has a kernel. Symmetric: scale = absmax / (2^(bits-1) - 1); asymmetric:
// fp16 scale and zero point over [min, max]. One thread per group.
__global__ void fakequant_rows_k(half* __restrict__ W, int I, int bits, int gs, int asym) {
    half* wr = W + (size_t)blockIdx.x * I;
    for (int g = threadIdx.x; g < I / gs; g += blockDim.x) {
        half* w = wr + g * gs;
        float lo = INFINITY, hi = -INFINITY;
        for (int i = 0; i < gs; i++) { const float v = __half2float(w[i]); lo = fminf(lo, v); hi = fmaxf(hi, v); }
        if (asym) {
            const int qm = (1 << bits) - 1;
            const float sc = __half2float(__float2half(hi > lo ? (hi - lo) / qm : 1.f)), z = rintf(-lo / sc);
            for (int i = 0; i < gs; i++) {
                const float q = fminf((float)qm, fmaxf(0.f, rintf(__half2float(w[i]) / sc) + z));
                w[i] = __float2half((q - z) * sc);
            }
        } else {
            const int qm = (1 << (bits - 1)) - 1;
            const float m = fmaxf(-lo, hi), sc = m > 0.f ? m / qm : 1.f;
            for (int i = 0; i < gs; i++)
                w[i] = __float2half(sc * fminf((float)qm, fmaxf((float)-(qm + 1), rintf(__half2float(w[i]) / sc))));
        }
    }
}

// A dense matrix in whichever form it was loaded: F16 (exact), int8 + row scales, or int4 gs64 + group scales
// (the experts' format: low nibble first, s is [O][I/64]).
struct Mat {
    const half* f16 = nullptr;
    const int8_t* q = nullptr;
    const float* s = nullptr;
    const uint8_t* q4 = nullptr;
    const uint8_t *q6 = nullptr, *q6h = nullptr;  // int6 gs64: low 4 bits (q4's nibble layout) and high 2 bits planes
};

template <int WARPS>
__global__ void gemv_q4_k(const uint8_t* __restrict__ Q, const float* __restrict__ s, const float* __restrict__ x,
                          float* __restrict__ y, int I, int O);
inline int persistent_blocks();
template <int WARPS, int NT>
__global__ void gemv_q6_k(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ Qh, const float* __restrict__ s,
                          const float* __restrict__ x, float* __restrict__ y, int I, int O);
template <int NT>
inline void gemv_q6(const Mat& m, const float* x, float* y, int I, int O, cudaStream_t st) {
    gemv_q6_k<8, NT><<<std::min(persistent_blocks(), (O + 7) / 8), 256, NT * (I + I / 8) * 4, st>>>(m.q6, m.q6h, m.s, x, y, I, O);
}

inline void gemv(const Mat& m, const float* x, float* y, int I, int O, cudaStream_t st = 0) {
    constexpr int WARPS = 8;
    if (m.q4) gemv_q4_k<WARPS><<<std::min(persistent_blocks(), (O + WARPS - 1) / WARPS), WARPS * 32, (I + I / 8) * 4, st>>>(m.q4, m.s, x, y, I, O);
    else if (m.q6) gemv_q6<1>(m, x, y, I, O, st);
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

// int8 weights, the common case: block tile 128 weight rows x 128 tokens x 32 K, 8 warps of 32 rows x 64 tokens,
// 16-byte loads, the next K step's loads in flight during the MMAs. Same operands as gemm_wmma_k (fp16, fp32 sums).
// Q6: W is the int6 low plane, Wh the high plane, s the group scales (applied while the tile is dequantized).
template <bool Q6 = false>
__global__ void __launch_bounds__(256) gemm_i8_k(const int8_t* __restrict__ W, const float* __restrict__ s,
                                                 const float* __restrict__ X, float* __restrict__ Y, int S, int I, int O,
                                                 const uint8_t* __restrict__ Wh = nullptr) {
    using namespace nvcuda;
    constexpr int BM = 128, BN = 128, BK = 32, LD = BK + 8;
    __shared__ __align__(32) half As[BM * LD], Bs[BN * LD];
    const int o0 = blockIdx.x * BM, s0 = blockIdx.y * BN, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 64;
    const int wr = tid >> 1, wc = (tid & 1) * 16;  // this thread's 16 weights of the tile
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][4];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) wmma::fill_fragment(c[i][j], 0.f);
    uint4 wv; float4 xv[4];
    uint2 lv; uint32_t hv; float sv;  // Q6: 16 low nibbles, 16 x 2 high bits, the group scale
    auto load = [&](int k0) {
        if constexpr (Q6) {
            const bool in = o0 + wr < O;
            const size_t r = (size_t)(o0 + wr), k = (size_t)(k0 + wc);
            lv = in ? *reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(W) + r * (I / 2) + k / 2) : make_uint2(0, 0);
            hv = in ? *reinterpret_cast<const uint32_t*>(Wh + r * (I / 4) + k / 4) : 0u;
            sv = in ? s[r * (I / 64) + k / 64] : 0.f;
        } else
        wv = o0 + wr < O ? *reinterpret_cast<const uint4*>(W + (size_t)(o0 + wr) * I + k0 + wc) : make_uint4(0, 0, 0, 0);
        for (int j = 0; j < 4; j++) {
            const int x = tid + 256 * j, r = x >> 3, k = (x & 7) * 4;
            xv[j] = s0 + r < S ? *reinterpret_cast<const float4*>(X + (size_t)(s0 + r) * I + k0 + k) : make_float4(0, 0, 0, 0);
        }
    };
    load(0);
    for (int k0 = 0; k0 < I; k0 += BK) {
        half2* a = reinterpret_cast<half2*>(As + wr * LD + wc);
        if constexpr (Q6) {
            const uint64_t lo = (uint64_t)lv.x | ((uint64_t)lv.y << 32);
            auto w6 = [&](int e) { return (float)((int)(((lo >> (4 * e)) & 15) | (((hv >> (2 * e)) & 3) << 4)) - 32) * sv; };
            for (int i = 0; i < 8; i++) a[i] = __floats2half2_rn(w6(2 * i), w6(2 * i + 1));
        } else {
            const int8_t* b = reinterpret_cast<const int8_t*>(&wv);
            for (int i = 0; i < 8; i++) a[i] = __halves2half2(__int2half_rn(b[2 * i]), __int2half_rn(b[2 * i + 1]));
        }
        for (int j = 0; j < 4; j++) {
            const int x = tid + 256 * j, r = x >> 3, k = (x & 7) * 4;
            half2* bp = reinterpret_cast<half2*>(Bs + r * LD + k);
            bp[0] = __floats2half2_rn(xv[j].x, xv[j].y); bp[1] = __floats2half2_rn(xv[j].z, xv[j].w);
        }
        __syncthreads();
        if (k0 + BK < I) load(k0 + BK);
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb[4];
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(fa[i], As + (wm + 16 * i) * LD + kk, LD);
            for (int j = 0; j < 4; j++) wmma::load_matrix_sync(fb[j], Bs + (wn + 16 * j) * LD + kk, LD);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) wmma::mma_sync(c[i][j], fa[i], fb[j], c[i][j]);
        }
        __syncthreads();
    }
    float* cs = reinterpret_cast<float*>(As) + warp * 256;  // the tiles are free now: a 16x16 scratch per warp
    for (int i = 0; i < 2; i++) for (int j = 0; j < 4; j++) {
        wmma::store_matrix_sync(cs, c[i][j], 16, wmma::mem_col_major);  // cs[token][row]: lanes along O (coalesced)
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int o = o0 + wm + 16 * i + (e & 15), t = s0 + wn + 16 * j + (e >> 4);
            if (o < O && t < S) Y[(size_t)t * O + o] = Q6 ? cs[e] : cs[e] * s[o];
        }
        __syncwarp();
    }
}

// Y[S][O] = X[S][I] . W^T for a dense matrix in either form.
inline void gemm(const Mat& m, const float* X, float* Y, int S, int I, int O, cudaStream_t st = 0) {
    if (S <= 2) {  // the MTP round: tensor-core tiles of 64 tokens would waste 62 of them; GEMV reads W once for both
        if (S == 2 && m.q) gemv_i8<8, 1, 2>(m.q, m.s, X, Y, I, O, st);
        else if (S == 2 && m.q6) gemv_q6<2>(m, X, Y, I, O, st);
        else for (int t = 0; t < S; t++) gemv(m, X + (size_t)t * I, Y + (size_t)t * O, I, O, st);
        return;
    }
    dim3 grid((O + 63) / 64, (S + 63) / 64);
    if (m.q4) gemm_wmma_k<uint8_t, true><<<grid, 128, 0, st>>>(m.q4, m.s, X, Y, S, I, O);
    else if (m.q) gemm_i8_k<<<dim3((O + 127) / 128, (S + 127) / 128), 256, 0, st>>>(m.q, m.s, X, Y, S, I, O);
    else if (m.q6) gemm_i8_k<true><<<dim3((O + 127) / 128, (S + 127) / 128), 256, 0, st>>>((const int8_t*)m.q6, m.s, X, Y, S, I, O, m.q6h);
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
    if (norm) {  // uniform: every thread reaches the barrier (a barrier inside `e < K` is undefined behavior)
        __syncthreads();  // thread 0's w[] writes are visible
        if (e < K) w[e] /= total;
    }
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

// ---- K11b: int6 gs64 dense GEMV (as K11, plus the high-bit plane: 8 bytes per 32 weights). NT vectors at once
// (the MTP round's 2 tokens: each weight read once). u = low nibble | high 2 bits << 4, weight = (u - 32) * scale.
__device__ __forceinline__ float dot32_q6(uint4 u, uint2 h, const float* __restrict__ x) {
    const uint32_t lo[4] = {u.x, u.y, u.z, u.w}, hw[4] = {h.x & 0xFFFFu, h.x >> 16, h.y & 0xFFFFu, h.y >> 16};
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float4 a = xv[2 * j], b = xv[2 * j + 1];
        const float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int n = 0; n < 8; n++)
            acc += (__int_as_float(0x4B000000u | ((lo[j] >> (4 * n)) & 0xFu) | (((hw[j] >> (2 * n)) & 3u) << 4)) - 8388640.f) * xs[n];
    }
    return acc;
}
template <int WARPS, int NT>
__global__ void gemv_q6_k(const uint8_t* __restrict__ Q, const uint8_t* __restrict__ Qh, const float* __restrict__ s,
                          const float* __restrict__ x, float* __restrict__ y, int I, int O) {
    extern __shared__ __align__(16) float xs[];  // NT x [xpad_size(I)]
    for (int t = 0; t < NT; t++)
        for (int i = threadIdx.x; i < I; i += blockDim.x) xs[t * xpad_size(I) + xpad(i)] = x[(size_t)t * I + i];
    __syncthreads();
    const int lane = threadIdx.x & 31, G = I / 64;
    for (int row = blockIdx.x * WARPS + (threadIdx.x >> 5); row < O; row += gridDim.x * WARPS) {
        const uint4* wr = reinterpret_cast<const uint4*>(Q + (size_t)row * (I / 2));
        const uint2* hr = reinterpret_cast<const uint2*>(Qh + (size_t)row * (I / 4));
        const float* sr = s + (size_t)row * G;
        float acc[NT] = {};
#pragma unroll 2
        for (int c = lane; c < I / 32; c += 32) {
            const uint4 u = __ldcs(wr + c);
            const uint2 h = __ldcs(hr + c);
            const float sc = sr[c >> 1];
#pragma unroll
            for (int t = 0; t < NT; t++) acc[t] += dot32_q6(u, h, xs + t * xpad_size(I) + 36 * c) * sc;
        }
#pragma unroll
        for (int t = 0; t < NT; t++) { const float v = warp_sum(acc[t]); if (lane == 0) y[(size_t)t * O + row] = v; }
    }
}
// F16 [O,I] -> int6 gs64 symmetric (scale = group absmax / 31, values in [-32, 31]; the same values as
// fakequant_rows_k(W, I, 6, 64, 0)). One thread per group.
__global__ void quantize_rows_q6_k(const half* __restrict__ W, uint8_t* __restrict__ Q, uint8_t* __restrict__ Qh,
                                   float* __restrict__ s, int I) {
    const half* wr = W + (size_t)blockIdx.x * I;
    for (int g = threadIdx.x; g < I / 64; g += blockDim.x) {
        float m = 0.f;
        for (int i = 0; i < 64; i++) m = fmaxf(m, fabsf(__half2float(wr[g * 64 + i])));
        const float sc = m > 0.f ? m / 31.f : 1.f;
        uint8_t* lo = Q + (size_t)blockIdx.x * (I / 2) + g * 32;
        uint8_t* hi = Qh + (size_t)blockIdx.x * (I / 4) + g * 16;
        int u[4];
        for (int i = 0; i < 64; i += 4) {
            for (int k = 0; k < 4; k++) u[k] = (int)fminf(31.f, fmaxf(-32.f, rintf(__half2float(wr[g * 64 + i + k]) / sc))) + 32;
            lo[i / 2] = (uint8_t)((u[0] & 15) | ((u[1] & 15) << 4));
            lo[i / 2 + 1] = (uint8_t)((u[2] & 15) | ((u[3] & 15) << 4));
            hi[i / 4] = (uint8_t)((u[0] >> 4) | ((u[1] >> 4) << 2) | ((u[2] >> 4) << 4) | ((u[3] >> 4) << 6));
        }
        s[(size_t)blockIdx.x * (I / 64) + g] = sc;
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
    if (g_skip) return;
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

// ---- PF1: the same prefill MoE on tensor cores (same work list, act and ypair as pairs_forward). A block
// takes 128 weight rows x 32 of an item's tokens; each K step is one int4 scale group (64 wide), dequantized
// to fp16 (v * scale, rounded once) in shared memory. Activations are rounded to fp16, accumulation is fp32.
// pairs_forward stays for the 2-token MTP round, where a GEMV reads each weight once for both rows.
constexpr int TC_BN = 32, TC_BK = 64, TC_LD = TC_BK + 8;  // +8 halves: no bank conflicts on fragment loads

// 32 int4 (low nibble first) times one scale -> 32 halves. Same int->float trick as dot32_i4.
__device__ __forceinline__ void dq32_i4(uint4 u, float s, half* __restrict__ dst) {
    const uint32_t q[4] = {u.x ^ 0x88888888u, u.y ^ 0x88888888u, u.z ^ 0x88888888u, u.w ^ 0x88888888u};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        half2* d2 = reinterpret_cast<half2*>(dst + 8 * j);
#pragma unroll
        for (int n = 0; n < 8; n += 2)
            d2[n / 2] = __floats2half2_rn((__int_as_float(0x4B000000u | ((q[j] >> (4 * n)) & 0xFu)) - 8388616.f) * s,
                                          (__int_as_float(0x4B000000u | ((q[j] >> (4 * n + 4)) & 0xFu)) - 8388616.f) * s);
    }
}

// C[128][16 * nj] (fp32, shared, aliasing As) = A rows . B^T over K = nk * TC_BK, nj = 1 or 2 token fragments.
// fetch(k0) loads one K step from global memory into the caller's registers, store() writes them (dequantized /
// rounded to fp16) into As[128][TC_LD] (row = local weight row) and Bs[TC_BN][TC_LD] (row = token). The next step's
// fetch is in flight while the tensor cores work on this one. 4 warps, 32 rows x 16 * nj tokens each.
template <class F, class S>
__device__ __forceinline__ void tc_tile(int nk, int nj, half* As, half* Bs, float* Cs, F fetch, S store) {
    using namespace nvcuda;
    const int warp = threadIdx.x >> 5;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
    for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::fill_fragment(c[i][j], 0.f);
    fetch(0);
    for (int kb = 0; kb < nk; kb++) {
        store();
        __syncthreads();
        if (kb + 1 < nk) fetch((kb + 1) * TC_BK);
#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], As + (warp * 32 + 16 * i) * TC_LD + kk, TC_LD);
            for (int j = 0; j < nj; j++) {
                wmma::load_matrix_sync(b, Bs + (16 * j) * TC_LD + kk, TC_LD);
                for (int i = 0; i < 2; i++) wmma::mma_sync(c[i][j], a[i], b, c[i][j]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++) for (int j = 0; j < nj; j++)
        wmma::store_matrix_sync(Cs + (warp * 32 + 16 * i) * TC_BN + 16 * j, c[i][j], TC_BN, wmma::mem_row_major);
    __syncthreads();
}

// Shared memory of one tc block: As + Bs during the K loop, then Cs (16 KB) over As. 23 KB: 4 blocks per SM.
struct TcSmem {
    __align__(32) half ab[128 * TC_LD + TC_BN * TC_LD];
    __device__ half* As() { return ab; }
    __device__ half* Bs() { return ab + 128 * TC_LD; }
    __device__ float* Cs() { return reinterpret_cast<float*>(ab); }
};
static_assert(128 * TC_LD * sizeof(half) >= 128 * TC_BN * sizeof(float), "Cs must fit over As");

// act[p][r0..r0+63] = silu(gate . x_t) * (up . x_t). Block (r0 / 64, item): local rows 0-63 gate, 64-127 up.
__global__ void __launch_bounds__(128) tc_gate_up_k(const PairItem* __restrict__ items, const Pair* __restrict__ pairs,
                                                    const float* __restrict__ X, float* __restrict__ act) {
    __shared__ TcSmem sm;
    __shared__ int tok[TC_BN], pid[TC_BN];
    const PairItem it = items[blockIdx.y];
    const int r0 = blockIdx.x * 64, tid = threadIdx.x;
    const float* sc = reinterpret_cast<const float*>(it.slot + EXPERT_Q_BYTES);
    half *As = sm.As(), *Bs = sm.Bs();
    for (int j0 = 0; j0 < it.count; j0 += TC_BN) {
        const int nb = min(TC_BN, it.count - j0), nj = nb > 16 ? 2 : 1;
        if (tid < TC_BN) { const Pair pr = pairs[it.start + j0 + min(tid, nb - 1)]; tok[tid] = pr.t; pid[tid] = pr.p; }
        __syncthreads();
        uint4 ua[2]; float sa[2], xb[TC_BN * TC_BK / 128];
        tc_tile(EH / TC_BK, nj, As, Bs, sm.Cs(),
            [&](int k0) {  // 128 rows x 64 = two uint4 per row; gate row g is merged row g, up row g is EI + g
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int e = tid + 128 * i, r = e >> 1, h = e & 1, gr = r < 64 ? r0 + r : EI + r0 + r - 64;
                    ua[i] = __ldg(reinterpret_cast<const uint4*>(it.slot + (size_t)gr * (EH / 2) + (k0 + 32 * h) / 2));
                    sa[i] = sc[gr * (EH / EGS) + k0 / EGS];
                }
#pragma unroll
                for (int i = 0; i < TC_BN * TC_BK / 128; i++) {
                    const int e = tid + 128 * i, n = e / TC_BK, k = e % TC_BK;
                    xb[i] = n < nb ? X[(size_t)tok[n] * EH + k0 + k] : 0.f;
                }
            },
            [&] {
#pragma unroll
                for (int i = 0; i < 2; i++) { const int e = tid + 128 * i; dq32_i4(ua[i], sa[i], As + (e >> 1) * TC_LD + 32 * (e & 1)); }
#pragma unroll
                for (int i = 0; i < TC_BN * TC_BK / 128; i++) {
                    const int e = tid + 128 * i;
                    Bs[(e / TC_BK) * TC_LD + e % TC_BK] = __float2half(xb[i]);
                }
            });
        const float* Cs = sm.Cs();
        for (int e = tid; e < 64 * TC_BN; e += 128) {
            const int r = e & 63, n = e >> 6;
            if (n < nb) {
                const float g = Cs[r * TC_BN + n], u = Cs[(64 + r) * TC_BN + n];
                act[(size_t)pid[n] * EI + r0 + r] = g / (1.f + __expf(-g)) * u;
            }
        }
        __syncthreads();
    }
}

// ypair[p][d0..d0+127] = w_p * (down . act_p). Block (d0 / 128, item).
__global__ void __launch_bounds__(128) tc_down_k(const PairItem* __restrict__ items, const Pair* __restrict__ pairs,
                                                 const float* __restrict__ act, float* __restrict__ ypair) {
    __shared__ TcSmem sm;
    __shared__ int pid[TC_BN];
    __shared__ float wt[TC_BN];
    const PairItem it = items[blockIdx.y];
    const int d0 = blockIdx.x * 128, tid = threadIdx.x;
    const uint8_t* dq = it.slot + (size_t)2 * EI * (EH / 2);
    const float* sc = reinterpret_cast<const float*>(it.slot + EXPERT_Q_BYTES) + 2 * EI * (EH / EGS);
    half *As = sm.As(), *Bs = sm.Bs();
    for (int j0 = 0; j0 < it.count; j0 += TC_BN) {
        const int nb = min(TC_BN, it.count - j0), nj = nb > 16 ? 2 : 1;
        if (tid < TC_BN) { const Pair pr = pairs[it.start + j0 + min(tid, nb - 1)]; pid[tid] = pr.p; wt[tid] = pr.w; }
        __syncthreads();
        uint4 ua[2]; float sa[2], xb[TC_BN * TC_BK / 128];
        tc_tile(EI / TC_BK, nj, As, Bs, sm.Cs(),
            [&](int k0) {
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const int e = tid + 128 * i, d = d0 + (e >> 1), h = e & 1;
                    ua[i] = __ldg(reinterpret_cast<const uint4*>(dq + (size_t)d * (EI / 2) + (k0 + 32 * h) / 2));
                    sa[i] = sc[d * (EI / EGS) + k0 / EGS];
                }
#pragma unroll
                for (int i = 0; i < TC_BN * TC_BK / 128; i++) {
                    const int e = tid + 128 * i, n = e / TC_BK, k = e % TC_BK;
                    xb[i] = n < nb ? act[(size_t)pid[n] * EI + k0 + k] : 0.f;
                }
            },
            [&] {
#pragma unroll
                for (int i = 0; i < 2; i++) { const int e = tid + 128 * i; dq32_i4(ua[i], sa[i], As + (e >> 1) * TC_LD + 32 * (e & 1)); }
#pragma unroll
                for (int i = 0; i < TC_BN * TC_BK / 128; i++) {
                    const int e = tid + 128 * i;
                    Bs[(e / TC_BK) * TC_LD + e % TC_BK] = __float2half(xb[i]);
                }
            });
        const float* Cs = sm.Cs();
        for (int e = tid; e < 128 * TC_BN; e += 128) {
            const int r = e & 127, n = e >> 7;
            if (n < nb) ypair[(size_t)pid[n] * EH + d0 + r] = wt[n] * Cs[r * TC_BN + n];
        }
        __syncthreads();
    }
}

inline void pairs_forward_tc(const PairItem* items, int n_items, const Pair* pairs, const float* X, float* act, float* ypair,
                             cudaStream_t st = 0) {
    if (!n_items) return;
    tc_gate_up_k<<<dim3(EI / 64, n_items), 128, 0, st>>>(items, pairs, X, act);
    tc_down_k<<<dim3(EH / 128, n_items), 128, 0, st>>>(items, pairs, act, ypair);
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

// Whole-step decode graph (one launch per token): the GPU hands each layer's routing to the host and waits for the
// host's CPU-expert result through mapped host memory flags instead of events and per-layer launches.
// export_signal_k: as export_route_k (one block), then flag = *seq once the data is visible to the host.
__global__ void export_signal_k(const float* __restrict__ h, const int* __restrict__ idx, const float* __restrict__ w,
                                float* h_out, int* idx_out, float* w_out, int D, int K, const int* __restrict__ seq, int* flag) {
    if (g_skip) return;
    for (int i = threadIdx.x; i < D; i += blockDim.x) h_out[i] = h[i];
    if (threadIdx.x < K) { idx_out[threadIdx.x] = idx[threadIdx.x]; w_out[threadIdx.x] = w[threadIdx.x]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile int*)flag = *seq;
}
// Spins until the host's flag = 2 * seq + (1 if it wrote a CPU result), then out += y (y: mapped host memory).
// Gives up after timeout_ns (a stalled host must not keep the GPU busy into a Windows TDR reset, 2 s): then y is not
// added, g_skip is set (layers l+1.. leave their state untouched) and *abort = seq tells the host
// to finish this token on the per-layer path. The last layer reports a normal pass in abort[1] = seq.
__global__ void wait_add_k(float* __restrict__ out, const float* y, const int* __restrict__ seq, const int* flag, int D,
                           int* abort, int l, int NL, long long timeout_ns,
                           long long* waited) {  // mapped host [NL]: ns this layer's GPU waited for the host (exposed CPU time)
    if (g_skip) return;
    __shared__ int f;
    if (threadIdx.x == 0) {
        const int want = *seq;
        long long t0, t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        int v;
        while (((v = *(const volatile int*)flag) >> 1) != want) {
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if (t - t0 > timeout_ns) {
                g_skip = 1;
                __threadfence_system();
                *(volatile int*)abort = want;
                v = 0;
                break;
            }
        }
        f = v;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        waited[l] = t - t0;
    }
    __syncthreads();
    if (f & 1)
        for (int i = threadIdx.x; i < D; i += blockDim.x) out[i] += ((const volatile float*)y)[i];
    if (l == NL - 1 && f && threadIdx.x == 0) { __threadfence_system(); *(volatile int*)(abort + 1) = *seq; }
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
    if (g_skip) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += sh[i] / (1.f + __expf(-gate_logit[0]));
}

// ---- K6: per-head RMSNorm(1+w) + partial RoPE, then append k/v to the cache ----
// q: [H][2*HD] (query ++ gate per head, from q_proj) -> query written to qo [H][HD].
// k, v: [KV][HD]. KV cache: int8 [KV][max_t][HD] + one f32 scale per (head, position), quantized when written
// (4x less VRAM than fp32: the difference goes to expert slots, and long contexts fit).
constexpr int AHD = 256;
struct KVCache { int8_t *k, *v; float *ks, *vs; };
__global__ void qk_norm_rope_k(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                               const half* __restrict__ qn, const half* __restrict__ kn,
                               float* __restrict__ qo, KVCache kv,
                               int H, int KV, const int* __restrict__ pos_d, int max_t, int rot, float theta, float eps, int ld) {
    if (g_skip) return;
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
    else {  // the whole block is a key head: quantize k and v of this position, one scale each
        const size_t row = (size_t)(hh - H) * max_t + pos, o = row * AHD + d;
        const float vv = v[(hh - H) * AHD + d];
        float mk = block_max(fabsf(x)), mv = block_max(fabsf(vv));
        float sk = mk > 0.f ? mk / 127.f : 1.f, sv = mv > 0.f ? mv / 127.f : 1.f;
        kv.k[o] = (int8_t)__float2int_rn(x / sk);
        kv.v[o] = (int8_t)__float2int_rn(vv / sv);
        if (d == 0) { kv.ks[row] = sk; kv.vs[row] = sv; }
    }
}

// ---- K7: decode attention (flash-decoding) ----
// ctx = softmax(q.K^T / sqrt(HD)) V over t in [0, T); out = ctx * sigmoid(gate).
// grid (H, tokens, ns): block s covers keys [s*L, (s+1)*L), L = ceil(T/ns), in tiles of ATILE scores with an online
// softmax, so shared memory is fixed and any context fits. ns > 1 (decode: H=16 blocks alone leave most SMs idle)
// writes partials {m, l, acc[HD]} to part, merged by attn_combine_k; ns == 1 (prefill chunks) writes out directly.
constexpr int ATILE = 2048;
__global__ void attn_decode_k(const float* __restrict__ qo, const float* __restrict__ q_gate,
                              KVCache kv, float* __restrict__ out, int H, int KV, const int* __restrict__ pos_d, int max_t, int ld,
                              float* __restrict__ part) {
    __shared__ float sc[ATILE];
    __shared__ __align__(16) float qs[AHD];
    const int tok = blockIdx.y, T = *pos_d + tok + 1;  // causal: token tok sees positions up to its own
    const int ns = gridDim.z, L = (T + ns - 1) / ns, lo = blockIdx.z * L, hi = min(T, lo + L);
    qo += (size_t)tok * H * AHD; q_gate += (size_t)tok * ld; out += (size_t)tok * H * AHD;
    int h = blockIdx.x, kvh = h / (H / KV), tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    qs[tid] = qo[h * AHD + tid];
    __syncthreads();
    const int8_t* K = kv.k + (size_t)kvh * max_t * AHD;
    const int8_t* V = kv.v + (size_t)kvh * max_t * AHD;
    const float* Ks = kv.ks + (size_t)kvh * max_t;
    const float* Vs = kv.vs + (size_t)kvh * max_t;
    const float scale = rsqrtf((float)AHD);
    float m = -INFINITY, l = 0.f, acc = 0.f;
    for (int t0 = lo; t0 < hi; t0 += ATILE) {
        const int n = min(ATILE, hi - t0);
        for (int i = wid; i < n; i += blockDim.x / 32) {  // warp per key, 8 dims per lane
            const int t = t0 + i;
            uint2 u = *reinterpret_cast<const uint2*>(K + (size_t)t * AHD + lane * 8);
            const float4* qr = reinterpret_cast<const float4*>(qs) + lane * 2;
            float4 qa = qr[0], qb = qr[1];
            float s = (float)(int8_t)(u.x) * qa.x + (float)(int8_t)(u.x >> 8) * qa.y + (float)(int8_t)(u.x >> 16) * qa.z + (float)(int8_t)(u.x >> 24) * qa.w +
                      (float)(int8_t)(u.y) * qb.x + (float)(int8_t)(u.y >> 8) * qb.y + (float)(int8_t)(u.y >> 16) * qb.z + (float)(int8_t)(u.y >> 24) * qb.w;
            s = warp_sum(s);
            if (lane == 0) sc[i] = s * Ks[t] * scale;
        }
        __syncthreads();
        float mt = -INFINITY;
        for (int i = tid; i < n; i += blockDim.x) mt = fmaxf(mt, sc[i]);
        const float mn = fmaxf(m, block_max(mt)), c = __expf(m - mn);  // m = -inf on the first tile: c = 0
        float s = 0.f;
        for (int i = tid; i < n; i += blockDim.x) { float e = __expf(sc[i] - mn); s += e; sc[i] = e * Vs[t0 + i]; }  // V's scale folded in
        l = l * c + block_sum(s);  // block_sum syncs, so sc[] is complete after it
        acc *= c;
        for (int i = 0; i < n; i++) acc += sc[i] * (float)V[(size_t)(t0 + i) * AHD + tid];
        m = mn;
        __syncthreads();  // sc[] is overwritten by the next tile
    }
    if (ns == 1) {
        float g = q_gate[h * 2 * AHD + AHD + tid];
        out[h * AHD + tid] = acc / l / (1.f + __expf(-g));
        return;
    }
    float* p = part + (((size_t)tok * H + h) * ns + blockIdx.z) * (AHD + 2);
    p[2 + tid] = acc;
    if (tid == 0) { p[0] = m; p[1] = l; }  // an empty split (T < ns) leaves m = -inf, l = 0, acc = 0
}

__global__ void attn_combine_k(const float* __restrict__ part, const float* __restrict__ q_gate,
                               float* __restrict__ out, int H, int ns, int ld) {
    const int tok = blockIdx.y, h = blockIdx.x, tid = threadIdx.x;
    const float* p = part + ((size_t)tok * H + h) * ns * (AHD + 2);
    float M = -INFINITY;
    for (int s = 0; s < ns; s++) M = fmaxf(M, p[s * (AHD + 2)]);  // split 0 is never empty, so M is finite
    float l = 0.f, acc = 0.f;
    for (int s = 0; s < ns; s++) {
        const float* q = p + s * (AHD + 2), w = __expf(q[0] - M);
        l += q[1] * w; acc += q[2 + tid] * w;
    }
    float g = q_gate[(size_t)tok * ld + h * 2 * AHD + AHD + tid];
    out[((size_t)tok * H + h) * AHD + tid] = acc / l / (1.f + __expf(-g));
}

// ---- K7P: prefill chunk attention (flash style, tensor cores, GQA) ----
// Block = (kv head, PR / G tokens): its G query heads x those tokens = PR query rows share every K/V tile of PK keys,
// read once. S = Q.K^T and O += P.V on wmma (fp16 operands: int8 K/V are exact; P carries V's row scale), online
// softmax in fp32, O in shared memory. Causal per row. Needs H / KV to divide PR.
constexpr int PR = 32, PK = 32, PLD = AHD + 8, PSL = PK + 4, PPL = PK + 8;
constexpr int ATTN_PF_SMEM = (PR + 2 * PK) * PLD * 2 + PR * PSL * 4 + PR * PPL * 2 + PR * PLD * 4 + 3 * PR * 4;
__global__ void __launch_bounds__(256) attn_prefill_k(const float* __restrict__ qo, const float* __restrict__ q_gate, KVCache kv,
                                                      float* __restrict__ out, int H, int KV, const int* __restrict__ pos_d,
                                                      int max_t, int ld, int T) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char sm[];
    half* Qs = reinterpret_cast<half*>(sm);     // [PR][PLD]
    half* Ks = Qs + PR * PLD;                    // [PK][PLD]
    half* Vs = Ks + PK * PLD;                    // [PK][PLD]
    float* Ss = reinterpret_cast<float*>(Vs + PK * PLD);  // [PR][PSL] scores
    half* Ps = reinterpret_cast<half*>(Ss + PR * PSL);    // [PR][PPL] probabilities * V scale
    float* Os = reinterpret_cast<float*>(Ps + PR * PPL);  // [PR][PLD]
    float *mrow = Os + PR * PLD, *lrow = mrow + PR, *crow = lrow + PR;
    const int G = H / KV, TT = PR / G, kvh = blockIdx.x, t0 = blockIdx.y * TT, tid = threadIdx.x, warp = tid >> 5;
    const int pos0 = *pos_d, nt = min(TT, T - t0), hi = pos0 + t0 + nt;  // keys [0, hi)
    for (int e = tid; e < PR * AHD; e += 256) {  // row r = token r / G, head kvh * G + r % G
        const int r = e / AHD, d = e % AHD, t = t0 + r / G;
        Qs[r * PLD + d] = __float2half(t < T ? qo[((size_t)t * H + kvh * G + r % G) * AHD + d] : 0.f);
        Os[r * PLD + d] = 0.f;
    }
    if (tid < PR) { mrow[tid] = -INFINITY; lrow[tid] = 0.f; }
    const int8_t *K = kv.k + (size_t)kvh * max_t * AHD, *V = kv.v + (size_t)kvh * max_t * AHD;
    const float *Ksc = kv.ks + (size_t)kvh * max_t, *Vsc = kv.vs + (size_t)kvh * max_t;
    const float scale = rsqrtf((float)AHD);
    const int r_sm = tid >> 3, q_sm = tid & 7;  // softmax: 8 threads per row, 4 keys each
    const int rpos = pos0 + t0 + min(r_sm / G, nt - 1);  // padding rows copy the last token's position
    for (int k0 = 0; k0 < hi; k0 += PK) {
        __syncthreads();  // the previous tile is done with Ks / Vs / Ps
        {
            const int key = tid >> 3, c = (tid & 7) * 32, t = k0 + key;
            uint4 a[2] = {}, b[2] = {};
            if (t < hi) for (int u = 0; u < 2; u++) {
                a[u] = reinterpret_cast<const uint4*>(K + (size_t)t * AHD + c)[u];
                b[u] = reinterpret_cast<const uint4*>(V + (size_t)t * AHD + c)[u];
            }
            const int8_t *ka = reinterpret_cast<const int8_t*>(a), *vb = reinterpret_cast<const int8_t*>(b);
            for (int i = 0; i < 32; i++) { Ks[key * PLD + c + i] = __int2half_rn(ka[i]); Vs[key * PLD + c + i] = __int2half_rn(vb[i]); }
        }
        __syncthreads();
        if (warp < 4) {  // S [PR][PK]: 2 x 2 fragments
            const int i = warp >> 1, j = warp & 1;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s;
            wmma::fill_fragment(s, 0.f);
            for (int kk = 0; kk < AHD; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, Qs + 16 * i * PLD + kk, PLD);
                wmma::load_matrix_sync(fb, Ks + 16 * j * PLD + kk, PLD);
                wmma::mma_sync(s, fa, fb, s);
            }
            wmma::store_matrix_sync(Ss + 16 * i * PSL + 16 * j, s, PSL, wmma::mem_row_major);
        }
        __syncthreads();
        {
            float sv[4], mx = -INFINITY;
            for (int u = 0; u < 4; u++) {
                const int kk = q_sm * 4 + u, t = k0 + kk;
                sv[u] = t <= rpos ? Ss[r_sm * PSL + kk] * Ksc[t] * scale : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mo = mrow[r_sm], mn = fmaxf(mo, mx), c = __expf(mo - mn);  // key 0 is in tile 0: mn is finite
            float sum = 0.f;
            for (int u = 0; u < 4; u++) {
                const int kk = q_sm * 4 + u, t = k0 + kk;
                const float p = __expf(sv[u] - mn);
                sum += p;
                Ps[r_sm * PPL + kk] = __float2half(t < hi ? p * Vsc[t] : 0.f);
            }
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (q_sm == 0) { mrow[r_sm] = mn; lrow[r_sm] = lrow[r_sm] * c + sum; crow[r_sm] = c; }
        }
        __syncthreads();
        for (int e = tid; e < PR * AHD; e += 256) Os[(e / AHD) * PLD + e % AHD] *= crow[e / AHD];
        __syncthreads();
        for (int i = 0; i < 2; i++)  // O [PR][AHD] += P . V: warp w owns column tiles 2w, 2w+1
            for (int j = 2 * warp; j < 2 * warp + 2; j++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> o;
                wmma::load_matrix_sync(o, Os + 16 * i * PLD + 16 * j, PLD, wmma::mem_row_major);
                for (int kk = 0; kk < PK; kk += 16) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
                    wmma::load_matrix_sync(fa, Ps + 16 * i * PPL + kk, PPL);
                    wmma::load_matrix_sync(fb, Vs + kk * PLD + 16 * j, PLD);
                    wmma::mma_sync(o, fa, fb, o);
                }
                wmma::store_matrix_sync(Os + 16 * i * PLD + 16 * j, o, PLD, wmma::mem_row_major);
            }
    }
    __syncthreads();
    for (int e = tid; e < PR * AHD; e += 256) {
        const int r = e / AHD, d = e % AHD, tl = r / G, t = t0 + tl, h = kvh * G + r % G;
        if (tl >= nt) continue;
        const float g = q_gate[(size_t)t * ld + h * 2 * AHD + AHD + d];
        out[((size_t)t * H + h) * AHD + d] = Os[r * PLD + d] / lrow[r] / (1.f + __expf(-g));
    }
}

inline void attn_prefill(const float* qo, const float* q, KVCache kv, float* out, int H, int KV, const int* pos_d, int max_t,
                         int ld, int T, cudaStream_t st) {
    static bool once = [] { CK(cudaFuncSetAttribute(attn_prefill_k, cudaFuncAttributeMaxDynamicSharedMemorySize, ATTN_PF_SMEM)); return true; }();
    (void)once;
    const int tt = PR / (H / KV);
    attn_prefill_k<<<dim3(KV, (T + tt - 1) / tt), 256, ATTN_PF_SMEM, st>>>(qo, q, kv, out, H, KV, pos_d, max_t, ld, T);
}

// ---- K7G: decode attention, GQA: one block per (kv head, token, split) for all 8 query heads of the group ----
// K and V are read once per kv head instead of once per query head. Keys in tiles of 256: thread i scores key i for
// all 8 heads, warp h runs head h's online softmax, thread d accumulates dim d of P.V for all 8 heads. Writes the same
// partials {m, l, acc[HD]} as attn_decode_k (merged by attn_combine_k). Needs H / KV == GQ.
constexpr int GQ = 8, GT = 256;
__global__ void __launch_bounds__(256) attn_gqa_k(const float* __restrict__ qo, KVCache kv, int H, int KV,
                                                  const int* __restrict__ pos_d, int max_t, float* __restrict__ part) {
    __shared__ __align__(16) float qs[GQ][AHD];  // the group's queries
    __shared__ __align__(16) float ps[GT][GQ];   // scores, then probabilities * V scale, [key][head]
    __shared__ float cs[GQ], ms[GQ], ls[GQ];
    const int tok = blockIdx.y, T = *pos_d + tok + 1, ns = gridDim.z, L = (T + ns - 1) / ns;
    const int lo = blockIdx.z * L, hi = min(T, lo + L), kvh = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    for (int h = 0; h < GQ; h++) qs[h][tid] = qo[((size_t)tok * H + kvh * GQ + h) * AHD + tid];
    const int8_t *K = kv.k + (size_t)kvh * max_t * AHD, *V = kv.v + (size_t)kvh * max_t * AHD;
    const float *Ks = kv.ks + (size_t)kvh * max_t, *Vs = kv.vs + (size_t)kvh * max_t;
    const float scale = rsqrtf((float)AHD);
    float acc[GQ] = {}, m = -INFINITY, l = 0.f;  // m, l: head wid's, the same in every lane
    __syncthreads();
    for (int t0 = lo; t0 < hi; t0 += GT) {
        const int n = min(GT, hi - t0);
        if (tid < n) {
            const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t)(t0 + tid) * AHD);
            float s[GQ] = {};
            for (int c = 0; c < AHD / 16; c++) {
                const uint4 u = kr[c];
                const int8_t* b = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
                for (int j = 0; j < 16; j += 4)
#pragma unroll
                    for (int h = 0; h < GQ; h++) {
                        const float4 qv = *reinterpret_cast<const float4*>(&qs[h][c * 16 + j]);
                        s[h] += b[j] * qv.x + b[j + 1] * qv.y + b[j + 2] * qv.z + b[j + 3] * qv.w;
                    }
            }
            const float sc = Ks[t0 + tid] * scale;
            for (int h = 0; h < GQ; h++) ps[tid][h] = s[h] * sc;
        }
        __syncthreads();
        {
            float mx = -INFINITY;
            for (int i = lane; i < n; i += 32) mx = fmaxf(mx, ps[i][wid]);
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mn = fmaxf(m, mx), c = __expf(m - mn);  // m = -inf on the first tile: c = 0
            float sum = 0.f;
            for (int i = lane; i < n; i += 32) { const float e = __expf(ps[i][wid] - mn); sum += e; ps[i][wid] = e * Vs[t0 + i]; }
            l = l * c + warp_sum(sum);
            m = mn;
            if (lane == 0) cs[wid] = c;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < GQ; h++) acc[h] *= cs[h];
        for (int i = 0; i < n; i++) {
            const float v = (float)V[(size_t)(t0 + i) * AHD + tid];
            const float4 p0 = *reinterpret_cast<const float4*>(&ps[i][0]), p1 = *reinterpret_cast<const float4*>(&ps[i][4]);
            acc[0] += p0.x * v; acc[1] += p0.y * v; acc[2] += p0.z * v; acc[3] += p0.w * v;
            acc[4] += p1.x * v; acc[5] += p1.y * v; acc[6] += p1.z * v; acc[7] += p1.w * v;
        }
        __syncthreads();  // ps is overwritten by the next tile
    }
    if (lane == 0) { ms[wid] = m; ls[wid] = l; }
    __syncthreads();
    for (int h = 0; h < GQ; h++) {  // an empty split (T < ns) leaves m = -inf, l = 0, acc = 0
        float* p = part + (((size_t)tok * H + kvh * GQ + h) * ns + blockIdx.z) * (AHD + 2);
        p[2 + tid] = acc[h];
        if (tid == 0) { p[0] = ms[h]; p[1] = ls[h]; }
    }
}

// pos_d: device int, the position of the first token, so the same launch can be replayed from a graph at any position.
// T tokens (a prefill chunk): q/k/v rows have stride ld; all K/V are appended before any token attends.
// ns > 1 needs part = T * H * ns * (AHD + 2) floats. Chunks of > 2 rows: attn_prefill_k (the MTP round's 2 rows and
// decode keep attn_decode_k).
inline void attn_decode(const float* q, const float* k, const float* v, const half* qn, const half* kn,
                        float* qo, KVCache kv, float* out, int H, int KV, const int* pos_d, int max_t,
                        int rot, float theta, float eps, int T, int ld, cudaStream_t st = 0, float* part = nullptr, int ns = 1) {
    qk_norm_rope_k<<<dim3(H + KV, T), AHD, 0, st>>>(q, k, v, qn, kn, qo, kv, H, KV, pos_d, max_t, rot, theta, eps, ld);
    if (T > 2 && ns == 1 && PR % (H / KV) == 0) { attn_prefill(qo, q, kv, out, H, KV, pos_d, max_t, ld, T, st); return; }
    if (ns > 1 && H / KV == GQ) {
        attn_gqa_k<<<dim3(KV, T, ns), GT, 0, st>>>(qo, kv, H, KV, pos_d, max_t, part);
        attn_combine_k<<<dim3(H, T), AHD, 0, st>>>(part, q, out, H, ns, ld);
        return;
    }
    attn_decode_k<<<dim3(H, T, ns), AHD, 0, st>>>(qo, q, kv, out, H, KV, pos_d, max_t, ld, part);
    if (ns > 1) attn_combine_k<<<dim3(H, T), AHD, 0, st>>>(part, q, out, H, ns, ld);
}

inline KVCache kv_alloc(int KV, int max_t) {
    KVCache c;
    CK(cudaMalloc(&c.k, (size_t)KV * max_t * AHD)); CK(cudaMalloc(&c.v, (size_t)KV * max_t * AHD));
    CK(cudaMalloc(&c.ks, (size_t)KV * max_t * 4)); CK(cudaMalloc(&c.vs, (size_t)KV * max_t * 4));
    return c;
}

__global__ void set_int_k(int* p, int v) { *p = v; }

// ---- K8: Gated DeltaNet decode ----
// Conv: causal depthwise conv (kernel CK) over C channels with a carried ring [C][CK-1], then SiLU.
// T tokens in order (T = 1 for decode, a chunk for prefill). qkv rows have stride ld; out is [T][C].
// ring0 (optional, T > 1): the ring after the first token, so the MTP round can drop the second (a wrong draft)
__global__ void dn_conv_k(const float* __restrict__ qkv, int ld, const half* __restrict__ w, float* __restrict__ ring,
                          float* __restrict__ out, int C, int CKn, int T, float* __restrict__ ring0 = nullptr) {
    if (g_skip) return;
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
        if (t == 0 && ring0) for (int i = 0; i < CKn - 1; i++) ring0[(size_t)c * (CKn - 1) + i] = rg[i];
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
                           int VH, int KH, float eps, int T, float* __restrict__ S0 = nullptr) {  // S0: state after token 0
    __shared__ float qs[DNK], ks[DNK];
    const int h = blockIdx.x, v = threadIdx.x, kh = h / (VH / KH);
    const float alog = expf(__half2float(A_log[h])), dtb = __half2float(dt_bias[h]), nwv = __half2float(nw[v]);
    float* Sh = S + (size_t)h * DNK * DNV;
    float col[DNK];
#pragma unroll
    for (int i = 0; i < DNK; i++) col[i] = Sh[i * DNV + v];
    for (int t = 0; t < T; t++) {
    if (g_skip) return;
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
        if (t == 0 && S0) {
#pragma unroll
            for (int i = 0; i < DNK; i++) S0[(size_t)h * DNK * DNV + i * DNV + v] = col[i];
        }
    }
#pragma unroll
    for (int i = 0; i < DNK; i++) Sh[i * DNV + v] = col[i];
}

// T tokens through conv + recurrence. qkv, z, a, b: rows of stride ld (one fused projection row per token).
inline void deltanet_core(const float* qkv, const float* z, const float* a, const float* b, int ld,
                          const half* conv_w, const half* A_log, const half* dt_bias, const half* nw,
                          float* ring, float* S, float* conv_scratch, float* o,
                          int C, int CKn, int VH, int KH, float eps, int T = 1, cudaStream_t st = 0,
                          float* S0 = nullptr, float* ring0 = nullptr) {
    dn_conv_k<<<(C + 255) / 256, 256, 0, st>>>(qkv, ld, conv_w, ring, conv_scratch, C, CKn, T, ring0);
    dn_recur_k<<<VH, DNV, 0, st>>>(conv_scratch, C, z, a, b, ld, A_log, dt_bias, nw, S, o, VH, KH, eps, T, S0);
}

// ---- K19: sampling helpers ----
// Repetition penalty (llama.cpp semantics) on the logits of recent distinct tokens, passed by value.
struct TokList { static constexpr int N = 64; int n = 0; int id[N]; };
__global__ void repeat_penalty_k(float* __restrict__ logits, TokList h, float pen) {
    int i = threadIdx.x;
    if (i < h.n) { float l = logits[h.id[i]]; logits[h.id[i]] = l > 0.f ? l / pen : l * pen; }
}
// Candidates for sampling: every logit within `cut` of the maximum (*maxv), compacted (unordered).
struct Cand { int id; float v; };
__global__ void candidates_k(const float* __restrict__ logits, int n, const float* __restrict__ maxv, float cut,
                             Cand* __restrict__ out, int* __restrict__ count, int cap) {
    const float thr = *maxv - cut;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = logits[i];
        if (v >= thr) { int k = atomicAdd(count, 1); if (k < cap) out[k] = {i, v}; }
    }
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
// gate_logit: the shared expert gate's logit, already computed (decode: with the router)
inline void shared_expert_add(const Mat& Wgu, const Mat& Wd, const float* gate_logit,
                              const float* x, float* out, float* scratch, int H, int SI, cudaStream_t st = 0) {
    float *g = scratch, *u = g + SI, *sh = u + SI;
    gemv(Wgu, x, g, H, 2 * SI, st);  // fused [gate; up] -> g, u
    silu_mul_k<<<(SI + 255) / 256, 256, 0, st>>>(g, u, SI);
    gemv(Wd, g, sh, SI, H, st);
    gated_add_k<<<(H + 255) / 256, 256, 0, st>>>(out, sh, gate_logit, H);
}
