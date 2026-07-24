// MoEx clean-room. Device kernels for batch-1 Qwen3-MoE decode.
//
// All quantized weights stay in canonical GGUF blocks in VRAM; kernels
// dequantize inline (design doc B2 / §10 kernel policy). The per-superblock
// decode math is the same validated formula proven bit-exact against the numpy
// oracle on the host (see moex-dequant-validated) — ported to __device__.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moex {

// Current launch stream for graph capture (0 = default stream). Set by
// Forward::step around a decode body so gemv/mmvq helpers join the capture.
inline cudaStream_t& moex_launch_stream() {
  static cudaStream_t s = 0;
  return s;
}

// ---- fp16 -> f32 (device) --------------------------------------------------
__device__ __forceinline__ float half_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -1;
      uint32_t m = mant;
      do { m <<= 1; ++e; } while ((m & 0x400) == 0);
      m &= 0x3ff;
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  return __int_as_float((int)bits);
}

__device__ __forceinline__ float rd_h(const uint8_t* p) {
  uint16_t h;
  h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  return half_to_f32(h);
}

// GGUF K-quant block byte sizes (verified vs published static_asserts).
enum : int {
  QK_K = 256,
  BQ8_0 = 34,   // 2 (d) + 32 (int8)
  BQ4_K = 144,  // 4 (d,dmin) + 12 (scales) + 128 (4-bit)
  BQ5_K = 176,  // 4 + 12 + 32 (qh) + 128 (ql)
  BQ6_K = 210,  // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q,
                                              uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Dequantize one contiguous 256-element superblock of the given type into out[256].
// `blk` points at the superblock's first byte.
__device__ inline void deq_superblock_q4k(const uint8_t* p, float* out) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* q = p + 16;
  int is = 0;
  uint8_t sc, m;
  float* y = out;
  for (int j = 0; j < 256; j += 64) {
    scale_min_k4(is + 0, scales, &sc, &m);
    const float d1 = d * sc, m1 = dmin * m;
    scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = dmin * m;
    for (int l = 0; l < 32; ++l) y[l] = d1 * (q[l] & 0x0F) - m1;
    for (int l = 0; l < 32; ++l) y[32 + l] = d2 * (q[l] >> 4) - m2;
    y += 64;
    q += 32;
    is += 2;
  }
}

__device__ inline void deq_superblock_q5k(const uint8_t* p, float* out) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* qh = p + 16;
  const uint8_t* ql = p + 48;
  int is = 0;
  uint8_t sc, m, u1 = 1, u2 = 2;
  float* y = out;
  for (int j = 0; j < 256; j += 64) {
    scale_min_k4(is + 0, scales, &sc, &m);
    const float d1 = d * sc, m1 = dmin * m;
    scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = dmin * m;
    for (int l = 0; l < 32; ++l)
      y[l] = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
    for (int l = 0; l < 32; ++l)
      y[32 + l] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
    y += 64;
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
}

__device__ inline void deq_superblock_q6k(const uint8_t* p, float* out) {
  const uint8_t* ql = p;
  const uint8_t* qh = p + 128;
  const int8_t* sc = (const int8_t*)(p + 192);
  const float d = rd_h(p + 208);
  float* yy = out;
  for (int nn = 0; nn < 256; nn += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      yy[l + 0] = d * sc[is + 0] * q1;
      yy[l + 32] = d * sc[is + 2] * q2;
      yy[l + 64] = d * sc[is + 4] * q3;
      yy[l + 96] = d * sc[is + 6] * q4;
    }
    yy += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

// ---- FUSED dequant+dot: decode each weight and MAC into a register, no buffer.
// Returns sum over the 256 superblock weights of w[i]*x[i]. This is the hot path
// for batch-1 decode; avoiding a float[256] scratch keeps everything in regs.
__device__ __forceinline__ float dot_superblock_q4k(const uint8_t* p, const float* x) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* q = p + 16;
  int is = 0;
  uint8_t sc, m;
  float acc = 0.0f;
  const float* xx = x;
  for (int j = 0; j < 256; j += 64) {
    scale_min_k4(is + 0, scales, &sc, &m);
    const float d1 = d * sc, m1 = dmin * m;
    scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = dmin * m;
    for (int l = 0; l < 32; ++l) acc += (d1 * (q[l] & 0x0F) - m1) * xx[l];
    for (int l = 0; l < 32; ++l) acc += (d2 * (q[l] >> 4) - m2) * xx[32 + l];
    xx += 64;
    q += 32;
    is += 2;
  }
  return acc;
}

__device__ __forceinline__ float dot_superblock_q5k(const uint8_t* p, const float* x) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* qh = p + 16;
  const uint8_t* ql = p + 48;
  int is = 0;
  uint8_t sc, m, u1 = 1, u2 = 2;
  float acc = 0.0f;
  const float* xx = x;
  for (int j = 0; j < 256; j += 64) {
    scale_min_k4(is + 0, scales, &sc, &m);
    const float d1 = d * sc, m1 = dmin * m;
    scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = dmin * m;
    for (int l = 0; l < 32; ++l)
      acc += (d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1) * xx[l];
    for (int l = 0; l < 32; ++l)
      acc += (d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2) * xx[32 + l];
    xx += 64;
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
  return acc;
}

__device__ __forceinline__ float dot_superblock_q6k(const uint8_t* p, const float* x) {
  const uint8_t* ql = p;
  const uint8_t* qh = p + 128;
  const int8_t* sc = (const int8_t*)(p + 192);
  const float d = rd_h(p + 208);
  float acc = 0.0f;
  const float* xx = x;
  for (int nn = 0; nn < 256; nn += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = (int)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = (int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      acc += d * sc[is + 0] * q1 * xx[l + 0];
      acc += d * sc[is + 2] * q2 * xx[l + 32];
      acc += d * sc[is + 4] * q3 * xx[l + 64];
      acc += d * sc[is + 6] * q4 * xx[l + 96];
    }
    xx += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
  return acc;
}

// ---- Lane-cooperative superblock dots: all 32 lanes of a warp work on ONE
// superblock together (lane = element position within each 32-elem group).
// The old whole-superblock-per-lane functions above idle 24-29 of 32 lanes on
// this model's 2048/768-column matrices (only nsb=8 or 3 superblocks per row);
// these keep every lane busy and make q-byte reads coalesced. The per-warp
// partial-sum association changes (fp reassociation within tolerance); the
// dequant value math is identical to the bit-exact-validated formulas above.
//
// Q4_K group map (matches dot_superblock_q4k): group g in 0..7 covers elems
// [32g, 32g+32); scale pair index = g; q byte = q[(g>>1)*32 + lane]; low
// nibble when g is even, high when odd.
__device__ __forceinline__ float dot_sb_q4k_lane(const uint8_t* p,
                                                 const float* x, int lane) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* q = p + 16;
  float acc = 0.0f;
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    uint8_t sc, m;
    scale_min_k4(g, scales, &sc, &m);
    const uint8_t b = q[(g >> 1) * 32 + lane];
    const int nib = (g & 1) ? (b >> 4) : (b & 0x0F);
    acc += (d * sc * nib - dmin * m) * x[g * 32 + lane];
  }
  return acc;
}

// ---- Q8 activation staging + dp4a dots (MMVQ-style, clean-room) ------------
// Activations are quantized once per GEMV block to int8 + per-32 scale, then
// Q4_K/Q6_K dots use __dp4a instead of float×dequant MACs.

__device__ __forceinline__ int moex_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  const int8_t* aa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* bb = reinterpret_cast<const int8_t*>(&b);
  return c + (int)aa[0] * bb[0] + (int)aa[1] * bb[1] + (int)aa[2] * bb[2] +
         (int)aa[3] * bb[3];
#endif
}

// Quantize float x[cols] (cols % 32 == 0) into q[cols] + d[cols/32] in smem.
__device__ __forceinline__ void quantize_activation_q8(const float* x, int cols,
                                                      float* d_out,
                                                      int8_t* q_out) {
  const int nblk = cols >> 5;
  for (int b = threadIdx.x; b < nblk; b += blockDim.x) {
    const float* xb = x + (b << 5);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float dv = amax / 127.0f + 1e-8f;
    const float id = 1.0f / dv;
    d_out[b] = dv;
    int8_t* qb = q_out + (b << 5);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      int q = __float2int_rn(xb[i] * id);
      q = q < -127 ? -127 : (q > 127 ? 127 : q);
      qb[i] = (int8_t)q;
    }
  }
}

__device__ __forceinline__ float warp_reduce_sum_f(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// One Q4_K superblock × Q8 acts (1 elem/lane). Reference / fallback.
__device__ __forceinline__ float dot_sb_q4k_q8_lane(const uint8_t* p,
                                                     const float* xd,
                                                     const int8_t* xq,
                                                     int lane) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* q = p + 16;
  float acc = 0.0f;
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    uint8_t sc, m;
    scale_min_k4(g, scales, &sc, &m);
    const uint8_t b = q[(g >> 1) * 32 + lane];
    const int nib = (g & 1) ? (b >> 4) : (b & 0x0F);
    const int q8 = (int)xq[g * 32 + lane];
    const float d8 = xd[g];
    acc += d * (float)sc * d8 * (float)(nib * q8) -
           dmin * (float)m * d8 * (float)q8;
  }
  return acc;
}

// Partial Q4_K×Q8_1 MMVQ-style dot for one superblock.
// iqs ∈ {0,2,...,30}: 16 threads cover all 256 elems via __dp4a.
// Clean-room (GGUF Q4_K qs layout + Q8 SoA scales); not a source copy.
__device__ __forceinline__ float vec_dot_q4k_q8_iqs(const uint8_t* p,
                                                     const float* xd,
                                                     const int8_t* xq,
                                                     int iqs) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* qs = p + 16;
  // 4 thread-pairs hit bq8_offset ∈ {0,2,4,6}; each covers two Q8 groups.
  const int bq8_offset = 2 * ((iqs / 2) / 4);
  const int* q4 = (const int*)(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int v0 = q4[0];
  const int v1 = q4[4];
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int g = bq8_offset + i;
    uint8_t sc, m;
    scale_min_k4(g, scales, &sc, &m);
    const float d8 = xd[g];
    const int* q8 = (const int*)(xq + g * 32) + ((iqs / 2) % 4);
    const int u0 = q8[0];
    const int u1 = q8[4];
    const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 = moex_dp4a(v1i, u1, moex_dp4a(v0i, u0, 0));
    const int dot2 =
        moex_dp4a(0x01010101, u1, moex_dp4a(0x01010101, u0, 0));
    sumf_d += d8 * (float)(dot1 * (int)sc);
    sumf_m += d8 * (float)(dot2 * (int)m);
  }
  return d * sumf_d - dmin * sumf_m;
}

// Full-warp Q4_K×Q8 over nsb superblocks (cols = 256*nsb). Lanes 0..15 and
// 16..31 stride odd/even superblocks (llama.cpp MMVQ 1-warp pattern).
__device__ __forceinline__ float row_dot_q4k_q8_mmvq(const uint8_t* row,
                                                      const float* xd,
                                                      const int8_t* xq,
                                                      int nsb, int lane) {
  float acc = 0.0f;
  const int iqs = 2 * (lane & 15);
#pragma unroll 1
  for (int sb = lane >> 4; sb < nsb; sb += 2) {
    acc += vec_dot_q4k_q8_iqs(row + (size_t)sb * BQ4_K, xd + sb * 8,
                              xq + sb * 256, iqs);
  }
  return acc;
}

// Q5_K: same grouping as Q4_K plus the high bit from qh[lane], mask 1<<g.
__device__ __forceinline__ float dot_sb_q5k_lane(const uint8_t* p,
                                                 const float* x, int lane) {
  const float d = rd_h(p);
  const float dmin = rd_h(p + 2);
  const uint8_t* scales = p + 4;
  const uint8_t* qh = p + 16;
  const uint8_t* ql = p + 48;
  const uint8_t hbyte = qh[lane];
  float acc = 0.0f;
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    uint8_t sc, m;
    scale_min_k4(g, scales, &sc, &m);
    const uint8_t b = ql[(g >> 1) * 32 + lane];
    const int nib = (g & 1) ? (b >> 4) : (b & 0x0F);
    const int hi = (hbyte & (1u << g)) ? 16 : 0;
    acc += (d * sc * (nib + hi) - dmin * m) * x[g * 32 + lane];
  }
  return acc;
}

// Q6_K: the reference inner loop is already elementwise in l — give each lane
// its own l and let it produce the same four outputs per 128-elem half.
__device__ __forceinline__ float dot_sb_q6k_lane(const uint8_t* p,
                                                 const float* x, int lane) {
  const uint8_t* ql = p;
  const uint8_t* qh = p + 128;
  const int8_t* sc = (const int8_t*)(p + 192);
  const float d = rd_h(p + 208);
  float acc = 0.0f;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint8_t* qlh = ql + 64 * half;
    const uint8_t* qhh = qh + 32 * half;
    const int8_t* sch = sc + 8 * half;
    const float* xx = x + 128 * half;
    const int is = lane >> 4;
    const int q1 = (int)((qlh[lane] & 0x0F) | (((qhh[lane] >> 0) & 3) << 4)) - 32;
    const int q2 = (int)((qlh[lane + 32] & 0x0F) | (((qhh[lane] >> 2) & 3) << 4)) - 32;
    const int q3 = (int)((qlh[lane] >> 4) | (((qhh[lane] >> 4) & 3) << 4)) - 32;
    const int q4 = (int)((qlh[lane + 32] >> 4) | (((qhh[lane] >> 6) & 3) << 4)) - 32;
    acc += d * sch[is + 0] * q1 * xx[lane + 0];
    acc += d * sch[is + 2] * q2 * xx[lane + 32];
    acc += d * sch[is + 4] * q3 * xx[lane + 64];
    acc += d * sch[is + 6] * q4 * xx[lane + 96];
  }
  return acc;
}

// Q6_K × Q8 acts with per-lane int products (fallback / reference).
__device__ __forceinline__ float dot_sb_q6k_q8_lane(const uint8_t* p,
                                                     const float* xd,
                                                     const int8_t* xq,
                                                     int lane) {
  const uint8_t* ql = p;
  const uint8_t* qh = p + 128;
  const int8_t* sc = (const int8_t*)(p + 192);
  const float d = rd_h(p + 208);
  float acc = 0.0f;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint8_t* qlh = ql + 64 * half;
    const uint8_t* qhh = qh + 32 * half;
    const int8_t* sch = sc + 8 * half;
    const int8_t* xqh = xq + 128 * half;
    const float* xdh = xd + 4 * half;  // 128 elems = 4 q8 blocks
    const int is = lane >> 4;
    const int q1 = (int)((qlh[lane] & 0x0F) | (((qhh[lane] >> 0) & 3) << 4)) - 32;
    const int q2 = (int)((qlh[lane + 32] & 0x0F) | (((qhh[lane] >> 2) & 3) << 4)) - 32;
    const int q3 = (int)((qlh[lane] >> 4) | (((qhh[lane] >> 4) & 3) << 4)) - 32;
    const int q4 = (int)((qlh[lane + 32] >> 4) | (((qhh[lane] >> 6) & 3) << 4)) - 32;
    const int8_t x1 = xqh[lane + 0];
    const int8_t x2 = xqh[lane + 32];
    const int8_t x3 = xqh[lane + 64];
    const int8_t x4 = xqh[lane + 96];
    acc += d * (float)sch[is + 0] * xdh[0] * (float)(q1 * (int)x1);
    acc += d * (float)sch[is + 2] * xdh[1] * (float)(q2 * (int)x2);
    acc += d * (float)sch[is + 4] * xdh[2] * (float)(q3 * (int)x3);
    acc += d * (float)sch[is + 6] * xdh[3] * (float)(q4 * (int)x4);
  }
  return acc;
}

__device__ __forceinline__ int moex_get_int_b2(const void* x, int i32) {
  const uint16_t* x16 = (const uint16_t*)x;
  return (int)x16[2 * i32] | ((int)x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ int moex_get_int_b4(const void* x, int i32) {
  return ((const int*)x)[i32];
}

// Partial Q6_K×Q8 MMVQ-style dot. iqs ∈ 0..31 (all warp lanes). Clean-room.
__device__ __forceinline__ float vec_dot_q6k_q8_iqs(const uint8_t* p,
                                                     const float* xd,
                                                     const int8_t* xq,
                                                     int iqs) {
  const uint8_t* ql = p;
  const uint8_t* qh = p + 128;
  const int8_t* sc_base = (const int8_t*)(p + 192);
  const float d = rd_h(p + 208);
  constexpr int QI6 = 32;  // QK_K/(4*QR6_K)
  const int bq8_offset =
      2 * 2 * (iqs / (QI6 / 2)) + (iqs % (QI6 / 2)) / (QI6 / 4);
  const int scale_offset =
      (QI6 / 4) * (iqs / (QI6 / 2)) + (iqs % (QI6 / 2)) / (QI6 / 8);
  const int vh_shift = 2 * ((iqs % (QI6 / 2)) / (QI6 / 4));
  const int vl = moex_get_int_b2(ql, iqs);
  const int vh =
      moex_get_int_b2(qh, (QI6 / 4) * (iqs / (QI6 / 2)) + iqs % (QI6 / 4)) >>
      vh_shift;
  const int8_t* scales = sc_base + scale_offset;
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int sc = (int)scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4(vil | vih, 0x20202020);
    const int q8_blk = bq8_offset + 2 * i;
    const int u = moex_get_int_b4(xq + q8_blk * 32, iqs % 8);
    const float d8 = xd[q8_blk];
    sumf += d8 * (float)(moex_dp4a(vi, u, 0) * sc);
  }
  return d * sumf;
}

__device__ __forceinline__ float row_dot_q6k_q8_mmvq(const uint8_t* row,
                                                      const float* xd,
                                                      const int8_t* xq,
                                                      int nsb, int lane) {
  float acc = 0.0f;
#pragma unroll 1
  for (int sb = 0; sb < nsb; ++sb) {
    acc += vec_dot_q6k_q8_iqs(row + (size_t)sb * BQ6_K, xd + sb * 8,
                              xq + sb * 256, lane);
  }
  return acc;
}

// GGML type codes we handle (mirror gguf::GgmlType values).
enum : int { GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14 };

}  // namespace moex
