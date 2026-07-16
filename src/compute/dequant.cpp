// MoEx clean-room K-quant dequantization. See dequant.h for layout notes.
#include "compute/dequant.h"

#include <cstring>

namespace moex {

float fp16_to_f32(uint16_t h) {
  // IEEE binary16 -> binary32, branch-light.
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      // subnormal: normalize
      int e = -1;
      uint32_t m = mant;
      do { m <<= 1; ++e; } while ((m & 0x400) == 0);
      m &= 0x3ff;
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);  // inf/nan
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

namespace {

inline float rd_fp16(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  return fp16_to_f32(h);
}

// Unpack the 6-bit scale and min for sub-block j from the packed 12 bytes.
inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

void dq_q8_0(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t b = 0; b < n / 32; ++b) {
    const uint8_t* p = src + b * 34;
    const float d = rd_fp16(p);
    const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
    for (int l = 0; l < 32; ++l) *y++ = d * q[l];
  }
}

void dq_q4_k(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t sb = 0; sb < n / 256; ++sb) {
    const uint8_t* p = src + sb * 144;
    const float d = rd_fp16(p);
    const float dmin = rd_fp16(p + 2);
    const uint8_t* scales = p + 4;
    const uint8_t* q = p + 16;
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
      scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * sc, m1 = dmin * m;
      scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * sc, m2 = dmin * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void dq_q5_k(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t sb = 0; sb < n / 256; ++sb) {
    const uint8_t* p = src + sb * 176;
    const float d = rd_fp16(p);
    const float dmin = rd_fp16(p + 2);
    const uint8_t* scales = p + 4;
    const uint8_t* qh = p + 16;
    const uint8_t* ql = p + 48;
    int is = 0;
    uint8_t sc, m, u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      scale_min_k4(is + 0, scales, &sc, &m);
      const float d1 = d * sc, m1 = dmin * m;
      scale_min_k4(is + 1, scales, &sc, &m);
      const float d2 = d * sc, m2 = dmin * m;
      for (int l = 0; l < 32; ++l)
        *y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l)
        *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
}

void dq_q6_k(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t sb = 0; sb < n / 256; ++sb) {
    const uint8_t* p = src + sb * 210;
    const uint8_t* ql = p;
    const uint8_t* qh = p + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
    const float d = rd_fp16(p + 208);
    float* yy = y;
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
    y += 256;
  }
}

void dq_f32(const uint8_t* src, float* y, uint64_t n) {
  std::memcpy(y, src, n * sizeof(float));
}

void dq_f16(const uint8_t* src, float* y, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i) y[i] = rd_fp16(src + i * 2);
}

}  // namespace

bool dequantize_row(gguf::GgmlType t, const uint8_t* src, float* dst, uint64_t n) {
  using T = gguf::GgmlType;
  switch (t) {
    case T::F32: dq_f32(src, dst, n); return true;
    case T::F16: dq_f16(src, dst, n); return true;
    case T::Q8_0: if (n % 32) return false; dq_q8_0(src, dst, n); return true;
    case T::Q4_K: if (n % 256) return false; dq_q4_k(src, dst, n); return true;
    case T::Q5_K: if (n % 256) return false; dq_q5_k(src, dst, n); return true;
    case T::Q6_K: if (n % 256) return false; dq_q6_k(src, dst, n); return true;
    default: return false;
  }
}

}  // namespace moex
