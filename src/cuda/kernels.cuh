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

// GGML type codes we handle (mirror gguf::GgmlType values).
enum : int { GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14 };

}  // namespace moex
