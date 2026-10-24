// GGUF (llama.cpp's model file) for the qwen35moe architecture (Qwen3.5/3.6-35B-A3B and its finetunes).
// `moex import FILE.gguf` converts it once into the moex container (int4 gs64 experts + F16 dense, config.json,
// tokenizer.json, mtp.safetensors), so the engine itself never reads GGUF and keeps every loading mode.
// Header-only and CUDA-free: tests/test_gguf.cpp builds it with cl alone.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "plat.h"
#include "json.h"
#include "gguf_iq.h"

enum GgType : uint32_t { GG_F32 = 0, GG_F16 = 1, GG_Q4_0 = 2, GG_Q4_1 = 3, GG_Q5_0 = 6, GG_Q5_1 = 7, GG_Q8_0 = 8, GG_Q2_K = 10,
                         GG_Q3_K = 11, GG_Q4_K = 12, GG_Q5_K = 13, GG_Q6_K = 14, GG_IQ3_XXS = 18, GG_IQ4_NL = 20, GG_IQ3_S = 21,
                         GG_IQ4_XS = 23, GG_BF16 = 30, GG_MXFP4 = 39 };

static std::string gg_type_name(uint32_t t) {
    static const char* n[] = {"F32", "F16", "Q4_0", "Q4_1", "4", "5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K",
                              "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S",
                              "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16"};
    return t < sizeof n / sizeof *n ? n[t] : t == 39 ? "MXFP4" : "type " + std::to_string(t);
}
static const char* GG_SUPPORTED = "F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, "
                                   "IQ3_XXS, IQ3_S, MXFP4";

// elements and bytes per block; false for a type moex does not read
static bool gg_block(uint32_t t, int& n, int& bytes) {
    switch (t) {
        case GG_F32: n = 1; bytes = 4; return true;
        case GG_F16: case GG_BF16: n = 1; bytes = 2; return true;
        case GG_Q8_0: n = 32; bytes = 34; return true;
        case GG_Q4_0: case GG_IQ4_NL: n = 32; bytes = 18; return true;
        case GG_Q4_1: n = 32; bytes = 20; return true;
        case GG_Q5_0: n = 32; bytes = 22; return true;
        case GG_Q5_1: n = 32; bytes = 24; return true;
        case GG_MXFP4: n = 32; bytes = 17; return true;
        case GG_Q2_K: n = 256; bytes = 84; return true;
        case GG_Q3_K: n = 256; bytes = 110; return true;
        case GG_IQ4_XS: n = 256; bytes = 136; return true;
        case GG_IQ3_XXS: n = 256; bytes = 98; return true;
        case GG_IQ3_S: n = 256; bytes = 110; return true;
        case GG_Q4_K: n = 256; bytes = 144; return true;
        case GG_Q5_K: n = 256; bytes = 176; return true;
        case GG_Q6_K: n = 256; bytes = 210; return true;
        default: return false;
    }
}

// IEEE half <-> float in plain C++ (nvcc's front end does not know the F16C intrinsics)
static inline float gg_half(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = h >> 10 & 0x1F, m = h & 0x3FF;
    uint32_t u;
    if (e == 0) { const float f = m * (1.0f / 16777216.0f); memcpy(&u, &f, 4); u |= s; }  // zero, subnormal: m * 2^-24
    else u = s | (e == 31 ? 0x7F800000 | m << 13 : (e + 112) << 23 | m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static inline float gg_h2f(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return gg_half(h); }
static inline uint16_t gg_f2h(float f) {  // round to nearest even
    uint32_t u; memcpy(&u, &f, 4);
    const uint16_t s = u >> 16 & 0x8000;
    u &= 0x7FFFFFFF;
    if (u > 0x7F800000) return s | 0x7E00;                         // NaN
    if (u >= 0x477FF000) return s | 0x7C00;                        // >= 65520: inf
    if (u < 0x38800000) { float a; memcpy(&a, &u, 4); return s | (uint16_t)std::nearbyint(a * 16777216.0f); }  // subnormal
    return s | (uint16_t)((u + 0xC8000FFF + (u >> 13 & 1)) >> 13);  // rebias the exponent, round the mantissa
}

static inline void gg_scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// Dequantize n elements (a whole number of blocks) at p, as llama.cpp's dequantize_row_* does.
static void gg_dequant(uint32_t t, const uint8_t* p, int64_t n, float* y) {
    switch (t) {
    case GG_F32: memcpy(y, p, n * 4); return;
    case GG_F16: for (int64_t i = 0; i < n; i++) y[i] = gg_h2f(p + 2 * i); return;
    case GG_BF16:
        for (int64_t i = 0; i < n; i++) { uint32_t u = (uint32_t)(p[2 * i] | p[2 * i + 1] << 8) << 16; memcpy(y + i, &u, 4); }
        return;
    case GG_Q8_0:
        for (int64_t b = 0; b < n / 32; b++, p += 34, y += 32) {
            const float d = gg_h2f(p);
            for (int i = 0; i < 32; i++) y[i] = d * (int8_t)p[2 + i];
        }
        return;
    case GG_Q4_K:
        for (int64_t b = 0; b < n / 256; b++, p += 144) {
            const float d = gg_h2f(p), dmin = gg_h2f(p + 2);
            const uint8_t *sc = p + 4, *q = p + 16;
            for (int j = 0, is = 0; j < 256; j += 64, q += 32, is += 2) {
                int s, m;
                gg_scale_min_k4(is, sc, s, m); const float d1 = d * s, m1 = dmin * m;
                gg_scale_min_k4(is + 1, sc, s, m); const float d2 = d * s, m2 = dmin * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            }
        }
        return;
    case GG_Q5_K:
        for (int64_t b = 0; b < n / 256; b++, p += 176) {
            const float d = gg_h2f(p), dmin = gg_h2f(p + 2);
            const uint8_t *sc = p + 4, *qh = p + 16, *ql = p + 48;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0, is = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
                int s, m;
                gg_scale_min_k4(is, sc, s, m); const float d1 = d * s, m1 = dmin * m;
                gg_scale_min_k4(is + 1, sc, s, m); const float d2 = d * s, m2 = dmin * m;
                for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
        }
        return;
    case GG_Q6_K:
        for (int64_t b = 0; b < n / 256; b++, p += 210) {
            const float d = gg_h2f(p + 208);
            const uint8_t *ql = p, *qh = p + 128;
            const int8_t* sc = (const int8_t*)(p + 192);
            for (int h = 0; h < 2; h++, y += 128, ql += 64, qh += 32, sc += 8)
                for (int l = 0; l < 32; l++) {
                    const int is = l / 16;
                    y[l] = d * sc[is] * (((ql[l] & 0xF) | ((qh[l] >> 0 & 3) << 4)) - 32);
                    y[l + 32] = d * sc[is + 2] * (((ql[l + 32] & 0xF) | ((qh[l] >> 2 & 3) << 4)) - 32);
                    y[l + 64] = d * sc[is + 4] * (((ql[l] >> 4) | ((qh[l] >> 4 & 3) << 4)) - 32);
                    y[l + 96] = d * sc[is + 6] * (((ql[l + 32] >> 4) | ((qh[l] >> 6 & 3) << 4)) - 32);
                }
        }
        return;
    case GG_Q4_0: case GG_Q4_1: case GG_Q5_0: case GG_Q5_1: case GG_IQ4_NL: case GG_MXFP4: {  // 32 per block
        static const int8_t kv_iq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        static const int8_t kv_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};  // E2M1, doubled
        int bn, bb;
        gg_block(t, bn, bb);
        for (int64_t b = 0; b < n / 32; b++, p += bb, y += 32) {
            float d, m = 0;
            const uint8_t* q;
            uint32_t qh = 0;
            if (t == GG_MXFP4) {  // E8M0 scale, halved for the doubled values
                const uint32_t bits = p[0] < 2 ? 0x00200000u << p[0] : (uint32_t)(p[0] - 1) << 23;
                memcpy(&d, &bits, 4); q = p + 1;
            } else {
                d = gg_h2f(p);
                q = p + (t == GG_Q4_1 || t == GG_Q5_1 ? 4 : 2);
                if (t == GG_Q4_1 || t == GG_Q5_1) m = gg_h2f(p + 2);
                if (t == GG_Q5_0 || t == GG_Q5_1) { memcpy(&qh, q, 4); q += 4; }
            }
            for (int j = 0; j < 16; j++) {  // element j in the low nibble of byte j, element j + 16 in the high one
                int lo = q[j] & 0xF, hi = q[j] >> 4;
                if (t == GG_Q5_0 || t == GG_Q5_1) { lo |= (qh >> j & 1) << 4; hi |= (qh >> (j + 16) & 1) << 4; }
                if (t == GG_IQ4_NL) { y[j] = d * kv_iq4[lo]; y[j + 16] = d * kv_iq4[hi]; }
                else if (t == GG_MXFP4) { y[j] = d * kv_fp4[lo]; y[j + 16] = d * kv_fp4[hi]; }
                else {
                    const int off = t == GG_Q4_0 ? 8 : t == GG_Q5_0 ? 16 : 0;
                    y[j] = d * (lo - off) + m; y[j + 16] = d * (hi - off) + m;
                }
            }
        }
        return;
    }
    case GG_IQ4_XS: {
        static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        for (int64_t b = 0; b < n / 256; b++, p += 136) {
            const float d = gg_h2f(p);
            const uint16_t sh = (uint16_t)(p[2] | p[3] << 8);
            const uint8_t *sl = p + 4, *q = p + 8;
            for (int ib = 0; ib < 8; ib++, q += 16, y += 32) {
                const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xF) | ((sh >> 2 * ib) & 3) << 4;
                const float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++) { y[j] = dl * kv[q[j] & 0xF]; y[j + 16] = dl * kv[q[j] >> 4]; }
            }
        }
        return;
    }
    case GG_Q2_K:
        for (int64_t b = 0; b < n / 256; b++, p += 84) {
            const uint8_t *sc = p, *q = p + 16;
            const float d = gg_h2f(p + 80), dmin = gg_h2f(p + 82);
            int is = 0;
            for (int h = 0; h < 2; h++, q += 32)
                for (int shift = 0; shift < 8; shift += 2)
                    for (int part = 0; part < 2; part++, is++) {
                        const float dl = d * (sc[is] & 0xF), ml = dmin * (sc[is] >> 4);
                        for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16 * part] >> shift) & 3) - ml;
                    }
        }
        return;
    case GG_Q3_K:
        for (int64_t b = 0; b < n / 256; b++, p += 110) {
            const uint8_t *hm = p, *q = p + 32;
            const float d = gg_h2f(p + 108);
            uint32_t aux[4];
            memcpy(aux, p + 96, 12);
            const uint32_t tmp = aux[2], k1 = 0x03030303, k2 = 0x0f0f0f0f;
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4);
            aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            int8_t sc[16];
            memcpy(sc, aux, 16);
            int is = 0;
            uint8_t m = 1;
            for (int h = 0; h < 2; h++, q += 32)
                for (int shift = 0; shift < 8; shift += 2, m <<= 1)
                    for (int part = 0; part < 2; part++, is++) {
                        const float dl = d * (sc[is] - 32);
                        for (int l = 0; l < 16; l++) {
                            const int k = l + 16 * part;
                            *y++ = dl * (((q[k] >> shift) & 3) - ((hm[k] & m) ? 0 : 4));
                        }
                    }
        }
        return;
    case GG_IQ3_XXS:
        for (int64_t b = 0; b < n / 256; b++, p += 98) {
            const float d = gg_h2f(p);
            const uint8_t *q = p + 2, *sa = p + 66;
            for (int ib = 0; ib < 8; ib++) {  // 32 values: 8 grid indices, a 4-bit scale and 4 x 7 sign bits
                uint32_t aux;
                memcpy(&aux, sa + 4 * ib, 4);
                const float db = d * (0.5f + (aux >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++) {
                    const uint8_t signs = iq_ksigns[(aux >> 7 * l) & 127];
                    const uint8_t *g1 = iq3xxs_grid[q[8 * ib + 2 * l]], *g2 = iq3xxs_grid[q[8 * ib + 2 * l + 1]];
                    for (int j = 0; j < 4; j++) *y++ = db * g1[j] * (signs >> j & 1 ? -1.f : 1.f);
                    for (int j = 0; j < 4; j++) *y++ = db * g2[j] * (signs >> (j + 4) & 1 ? -1.f : 1.f);
                }
            }
        }
        return;
    case GG_IQ3_S:
        for (int64_t b = 0; b < n / 256; b++, p += 110) {
            const float d = gg_h2f(p);
            const uint8_t *q = p + 2, *qh = p + 66, *sg = p + 74, *sc = p + 106;
            for (int i = 0; i < 64; i++) {  // grid index i covers elements 4i .. 4i + 3; one 4-bit scale per 32
                const float db = d * (1 + 2 * ((sc[i / 16] >> 4 * (i / 8 % 2)) & 0xF));
                const uint8_t* g = iq3s_grid[q[i] | ((qh[i / 8] >> (i % 8)) & 1) << 8];
                for (int j = 0; j < 4; j++) { const int e = 4 * i + j; *y++ = db * g[j] * (sg[e / 8] >> (e % 8) & 1 ? -1.f : 1.f); }
            }
        }
        return;
    }
    throw std::runtime_error("cannot dequantize " + gg_type_name(t));
}

// Experts are written in moex's zero-point form: unsigned nibbles u, w = s * u + m, one (s, m) per 32 inputs
// (kernels.cuh). One GGUF Q4_K superblock (256 weights, 8 sub-blocks of 32 with a 6-bit scale and min each) is
// exactly that form: s = d * scale, m = -dmin * min, u = the nibble. Q4_K keeps sub-block j's 32 values in the low
// nibbles of 32 bytes and sub-block j + 1's in the high ones; the element order is restored here.
static void gg_q4k_zp(const uint8_t* p, uint8_t* u, float* s, float* m) {
    const float d = gg_h2f(p), dmin = gg_h2f(p + 2);
    const uint8_t *sc = p + 4, *q = p + 16;
    for (int j = 0; j < 4; j++, q += 32) {
        int a, b;
        gg_scale_min_k4(2 * j, sc, a, b); s[2 * j] = d * a; m[2 * j] = -dmin * b;
        gg_scale_min_k4(2 * j + 1, sc, a, b); s[2 * j + 1] = d * a; m[2 * j + 1] = -dmin * b;
        for (int l = 0; l < 32; l++) { u[64 * j + l] = q[l] & 0xF; u[64 * j + 32 + l] = q[l] >> 4; }
    }
}

// Any other type: 32 weights -> u in [0, 15] and (s, m) by least squares. Start from the min/max grid, then twice:
// refit (s, m) for the rounding, re-round to the new grid. Never worse than plain min/max rounding (test_gguf).
static void gg_zp_group(const float* x, uint8_t* u, float& s, float& m) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < 32; i++) { lo = std::min(lo, x[i]); hi = std::max(hi, x[i]); }
    s = (hi - lo) / 15; m = lo;
    if (s == 0) { memset(u, 0, 32); return; }
    auto err = [&](float s_, float m_, const uint8_t* v) { float e = 0; for (int i = 0; i < 32; i++) { const float r = x[i] - s_ * v[i] - m_; e += r * r; } return e; };
    auto round_to = [&](float s_, float m_, uint8_t* v) { for (int i = 0; i < 32; i++) v[i] = (uint8_t)std::clamp((int)std::nearbyint((x[i] - m_) / s_), 0, 15); };
    round_to(s, m, u);
    float best = err(s, m, u);
    uint8_t v[32];
    float s1 = s, m1 = m;
    for (int it = 0; it < 2; it++) {
        double su = 0, suu = 0, sx = 0, sxu = 0;
        const uint8_t* cur = it ? v : u;
        for (int i = 0; i < 32; i++) { su += cur[i]; suu += cur[i] * cur[i]; sx += x[i]; sxu += (double)x[i] * cur[i]; }
        const double det = 32 * suu - su * su;
        if (det <= 0) break;
        s1 = (float)((32 * sxu - su * sx) / det); m1 = (float)((sx - s1 * su) / 32);
        if (s1 <= 0) break;
        round_to(s1, m1, v);
        const float e = err(s1, m1, v);
        if (e < best) { best = e; s = s1; m = m1; memcpy(u, v, 32); }
    }
}

// One row of a GGUF expert tensor (cols inputs) -> nibble bytes (element 2k low, 2k + 1 high) and half2 (s, m) per 32
static void gg_zp_row(uint32_t type, const uint8_t* src, int cols, uint8_t* qb, uint16_t* sm, float* tmp) {
    uint8_t u[256];
    float s[8], m[8];
    for (int b = 0; b < cols / 256; b++) {
        if (type == GG_Q4_K) gg_q4k_zp(src + (size_t)b * 144, u, s, m);
        else {
            int bn, bb;
            gg_block(type, bn, bb);
            gg_dequant(type, src + (size_t)b * 256 / bn * bb, 256, tmp);
            for (int g = 0; g < 8; g++) gg_zp_group(tmp + 32 * g, u + 32 * g, s[g], m[g]);
        }
        for (int i = 0; i < 128; i++) qb[b * 128 + i] = (uint8_t)(u[2 * i] | u[2 * i + 1] << 4);
        for (int g = 0; g < 8; g++) { sm[(b * 8 + g) * 2] = gg_f2h(s[g]); sm[(b * 8 + g) * 2 + 1] = gg_f2h(m[g]); }
    }
}

static void gg_par_for(int64_t n, const std::function<void(int64_t)>& f) {
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 0; i < nt; i++) th.emplace_back([&] { for (int64_t k; (k = next.fetch_add(1)) < n;) f(k); });
    for (auto& t : th) t.join();
}

struct Gguf {
    struct Kv { uint32_t type = 0; double num = 0; std::string str; std::vector<std::string> strs; std::vector<double> nums; };
    struct Tensor { std::vector<int64_t> ne; uint32_t type = 0; const uint8_t* data = nullptr; int64_t n = 1, row_bytes = 0; };
    std::map<std::string, Kv> kv;
    std::map<std::string, Tensor> t;
    const uint8_t* base = nullptr;
    uint64_t size = 0;

    explicit Gguf(const std::string& path) {
#ifdef _WIN32
        HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER sz; GetFileSizeEx(f, &sz); size = (uint64_t)sz.QuadPart;
        HANDLE m = size ? CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
        base = m ? (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
#else
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        size = std::filesystem::file_size(path);
        void* p = size ? mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
        base = p == MAP_FAILED ? nullptr : (const uint8_t*)p;
#endif
        if (!base) throw std::runtime_error("cannot map " + path);  // mapped for the process lifetime
        parse();
    }

    const Kv* get(const std::string& k) const { auto it = kv.find(k); return it == kv.end() ? nullptr : &it->second; }
    bool has(const std::string& name) const { return t.count(name) > 0; }
    const Tensor& at(const std::string& name) const {
        auto it = t.find(name);
        if (it == t.end()) throw std::runtime_error("tensor " + name + " is missing from the GGUF");
        return it->second;
    }
    // rows [r0, r0 + nr) of a tensor (a row = ne[0] elements) as f32
    void rows(const Tensor& x, int64_t r0, int64_t nr, float* y) const { gg_dequant(x.type, x.data + r0 * x.row_bytes, nr * x.ne[0], y); }
    std::vector<float> all(const std::string& name) const {
        const Tensor& x = at(name);
        std::vector<float> v(x.n);
        const int64_t nr = x.n / x.ne[0], chunk = 1024;
        gg_par_for((nr + chunk - 1) / chunk, [&](int64_t c) { rows(x, c * chunk, std::min(chunk, nr - c * chunk), v.data() + c * chunk * x.ne[0]); });
        return v;
    }

private:
    uint64_t pos = 0;
    const uint8_t* need(uint64_t n) {
        if (n > size || pos > size - n) throw std::runtime_error("GGUF header is truncated or corrupt");
        const uint8_t* p = base + pos; pos += n; return p;
    }
    template <class T> T rd() { T v; memcpy(&v, need(sizeof v), sizeof v); return v; }
    std::string rstr() { const uint64_t n = rd<uint64_t>(); return std::string((const char*)need(n), n); }
    double rnum(uint32_t ty) {
        switch (ty) {
            case 0: return rd<uint8_t>(); case 1: return rd<int8_t>(); case 2: return rd<uint16_t>(); case 3: return rd<int16_t>();
            case 4: return rd<uint32_t>(); case 5: return rd<int32_t>(); case 6: return rd<float>(); case 7: return rd<uint8_t>();
            case 10: return (double)rd<uint64_t>(); case 11: return (double)rd<int64_t>(); case 12: return rd<double>();
        }
        throw std::runtime_error("GGUF metadata has an unknown value type " + std::to_string(ty));
    }
    void parse() {
        if (size < 4 || memcmp(base, "GGUF", 4)) throw std::runtime_error("not a GGUF file (no GGUF magic)");
        pos = 4;
        const uint32_t ver = rd<uint32_t>();
        if (ver != 2 && ver != 3) throw std::runtime_error("GGUF version " + std::to_string(ver) + " (moex reads v2 and v3)");
        const uint64_t nt = rd<uint64_t>(), nkv = rd<uint64_t>();
        for (uint64_t i = 0; i < nkv; i++) {
            std::string k = rstr();
            Kv v; v.type = rd<uint32_t>();
            if (v.type == 8) v.str = rstr();
            else if (v.type == 9) {
                const uint32_t at = rd<uint32_t>(); const uint64_t n = rd<uint64_t>();
                if (n > size) throw std::runtime_error("GGUF header is corrupt");
                for (uint64_t j = 0; j < n; j++) { if (at == 8) v.strs.push_back(rstr()); else v.nums.push_back(rnum(at)); }
            } else v.num = rnum(v.type);
            kv[k] = std::move(v);
        }
        std::vector<std::pair<std::string, uint64_t>> offs;
        for (uint64_t i = 0; i < nt; i++) {
            std::string name = rstr();
            Tensor x;
            const uint32_t nd = rd<uint32_t>();
            if (nd < 1 || nd > 4) throw std::runtime_error("tensor " + name + " has " + std::to_string(nd) + " dimensions");
            for (uint32_t d = 0; d < nd; d++) { x.ne.push_back((int64_t)rd<uint64_t>()); x.n *= x.ne.back(); }
            x.type = rd<uint32_t>();
            offs.push_back({name, rd<uint64_t>()});
            t[name] = x;
        }
        const Kv* al = get("general.alignment");
        const uint64_t align = al ? (uint64_t)al->num : 32;
        const uint64_t start = (pos + align - 1) / align * align;
        for (auto& [name, off] : offs) {
            Tensor& x = t[name];
            int bn, bb;
            if (!gg_block(x.type, bn, bb)) continue;  // checked (with a clear message) only where moex needs it
            if (x.ne[0] % bn) throw std::runtime_error("tensor " + name + ": row length is not a whole number of blocks");
            x.row_bytes = x.ne[0] / bn * bb;
            const uint64_t bytes = (uint64_t)(x.n / x.ne[0]) * x.row_bytes;
            if (start + off < start || start + off > size || bytes > size - start - off)
                throw std::runtime_error("tensor " + name + " lies past the end of the file (incomplete download?)");
            x.data = base + start + off;
        }
    }
};

// ---- qwen35moe -> moex container ----------------------------------------------------------------------------------

// The container's config.json (the engine's shapes are fixed; the importer refuses any other shape).
static const char* GG_CONFIG_JSON = R"({
 "architectures": ["Qwen3_5MoeForCausalLM"],
 "attn_output_gate": true, "bos_token_id": 248044, "eos_token_id": 248044, "full_attention_interval": 4,
 "head_dim": 256, "hidden_act": "silu", "hidden_size": 2048, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
 "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_value_head_dim": 128, "model_type": "qwen3_5_moe_text",
 "moe_intermediate_size": 512, "mtp_num_hidden_layers": 1, "num_attention_heads": 16, "num_experts": 256,
 "num_experts_per_tok": 8, "num_hidden_layers": 40, "num_key_value_heads": 2, "partial_rotary_factor": 0.25,
 "rms_norm_eps": 1e-06, "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
 "rope_theta": 10000000, "rope_type": "default"}, "shared_expert_intermediate_size": 512, "tie_word_embeddings": false,
 "vocab_size": 248320, "moex_source": "%s"
}
)";

// tokenizer.json around the GGUF's vocab and merges: Qwen3.5's pre-tokenizer (GGUF `tokenizer.ggml.pre` = qwen35)
static const char* GG_TOK_MID = R"(  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
  "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
)";

static std::string gg_tokenizer_json(const Gguf& g) {
    const Gguf::Kv *tok = g.get("tokenizer.ggml.tokens"), *ty = g.get("tokenizer.ggml.token_type"), *mg = g.get("tokenizer.ggml.merges");
    if (!tok || !ty || !mg || tok->strs.size() != ty->nums.size()) throw std::runtime_error("the GGUF has no complete BPE tokenizer (tokens, token_type, merges)");
    std::string o = "{\n  \"version\": \"1.0\",\n  \"truncation\": null,\n  \"padding\": null,\n  \"added_tokens\": [";
    bool first = true;
    for (size_t i = 0; i < tok->strs.size(); i++) {
        const int tt = (int)ty->nums[i];  // 1 normal, 3 control (special), 4 user-defined (e.g. <think>), 5 unused padding
        if (tt != 3 && tt != 4) continue;
        o += std::string(first ? "\n" : ",\n") + "    {\"id\": " + std::to_string(i) + ", \"content\": \"" + jesc(tok->strs[i]) +
             "\", \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": " + (tt == 3 ? "true}" : "false}");
        first = false;
    }
    o += "\n  ],\n";
    o += GG_TOK_MID;
    o += "    \"vocab\": {";
    first = true;
    for (size_t i = 0; i < tok->strs.size(); i++)
        if ((int)ty->nums[i] == 1) { o += std::string(first ? "\n" : ",\n") + "      \"" + jesc(tok->strs[i]) + "\": " + std::to_string(i); first = false; }
    o += "\n    },\n    \"merges\": [";
    for (size_t i = 0; i < mg->strs.size(); i++) o += std::string(i ? ",\n" : "\n") + "      \"" + jesc(mg->strs[i]) + "\"";
    return o + "\n    ]\n  }\n}\n";
}

// safetensors writer: the header first (sizes are known up front), then each tensor from its fill callback
struct GgOut {
    std::string name, dtype;
    std::vector<int64_t> shape;
    uint64_t bytes;
    std::function<void(uint8_t*)> fill;
};
static void gg_write_st(const std::string& path, const std::vector<GgOut>& ts) {
    std::string h = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); i++) {
        h += (i ? ",\"" : "\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[";
        for (size_t d = 0; d < ts[i].shape.size(); d++) h += (d ? "," : "") + std::to_string(ts[i].shape[d]);
        h += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + ts[i].bytes) + "]}";
        off += ts[i].bytes;
    }
    h += "}";
    h.append((8 - h.size() % 8) % 8, ' ');
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const uint64_t n = h.size();
    bool ok = fwrite(&n, 8, 1, f) == 1 && fwrite(h.data(), 1, n, f) == n;
    std::vector<uint8_t> buf;
    for (auto& x : ts) {
        if (!ok) break;
        buf.resize(x.bytes);
        x.fill(buf.data());
        ok = fwrite(buf.data(), 1, x.bytes, f) == x.bytes;
    }
    if (fclose(f) || !ok) throw std::runtime_error("write failed: " + path + " (disk full?)");
}

static const int GG_D = 2048, GG_E = 256, GG_I = 512, GG_VH = 32, GG_HD = 128;
static const uint64_t GG_EXPERT_Q = 3ull * GG_I * GG_D / 2, GG_EXPERT_S = 3ull * GG_I * GG_D / 32 * 4;  // half2 per 32

// llama.cpp's converter stores Qwen3.5's DeltaNet value heads reordered: GGUF head j holds HF head 2 * (j % 16) + j / 16.
static inline int gg_vhead_src(int hf) { return (hf % 2) * 16 + hf / 2; }  // HF value head -> GGUF value head
// rows [r0, r0 + 32 * per) grouped by value head: GGUF order -> HF order
static void gg_unperm_rows(std::vector<float>& w, int64_t r0, int64_t per, int64_t cols) {
    std::vector<float> c(w.begin() + r0 * cols, w.begin() + (r0 + GG_VH * per) * cols);
    for (int h = 0; h < GG_VH; h++)
        memcpy(&w[(r0 + h * per) * cols], &c[gg_vhead_src(h) * per * cols], per * cols * 4);
}

// Validate the GGUF against the one shape the engine runs; throws with what differs.
static void gg_check_shape(const Gguf& g, bool& mtp) {
    const Gguf::Kv* a = g.get("general.architecture");
    if (!a || a->str != "qwen35moe")
        throw std::runtime_error("this GGUF's architecture is " + (a ? a->str : std::string("unknown")) +
                                 "; moex runs qwen35moe (Qwen3.5/3.6-35B-A3B and its finetunes)");
    auto num = [&](const std::string& k) -> double {
        const Gguf::Kv* v = g.get("qwen35moe." + k);
        if (!v) throw std::runtime_error("GGUF metadata qwen35moe." + k + " is missing");
        return v->num;
    };
    const double nx = g.get("qwen35moe.nextn_predict_layers") ? num("nextn_predict_layers") : 0;
    const double bc = num("block_count");
    if (!((bc == 40 && nx == 0) || (bc == 41 && nx == 1)))
        throw std::runtime_error("this GGUF has " + std::to_string((int)bc) + " blocks (nextn " + std::to_string((int)nx) +
                                 "); moex runs the 40-layer 35B-A3B (41 with its MTP layer)");
    mtp = nx == 1;
    const std::pair<const char*, double> want[] = {
        {"embedding_length", 2048}, {"attention.head_count", 16}, {"attention.head_count_kv", 2}, {"attention.key_length", 256},
        {"attention.value_length", 256}, {"expert_count", 256}, {"expert_used_count", 8}, {"expert_feed_forward_length", 512},
        {"expert_shared_feed_forward_length", 512}, {"ssm.conv_kernel", 4}, {"ssm.state_size", 128}, {"ssm.group_count", 16},
        {"ssm.time_step_rank", 32}, {"ssm.inner_size", 4096}, {"full_attention_interval", 4}, {"rope.freq_base", 1e7}};
    for (auto& [k, v] : want)
        if (num(k) != v) throw std::runtime_error(std::string("this GGUF has qwen35moe.") + k + " = " + std::to_string(num(k)) +
                                                  "; moex runs only the 35B-A3B shape (" + std::to_string(v) + ")");
    const double eps = num("attention.layer_norm_rms_epsilon");
    if (std::fabs(eps - 1e-6) > 1e-8) throw std::runtime_error("this GGUF's RMSNorm epsilon is " + std::to_string(eps) + "; moex uses 1e-6");
    const Gguf::Kv* rs = g.get("qwen35moe.rope.dimension_sections");
    if (!rs || rs->nums.size() < 3 || rs->nums[0] != 11 || rs->nums[1] != 11 || rs->nums[2] != 10 || (rs->nums.size() > 3 && rs->nums[3] != 0))
        throw std::runtime_error("this GGUF's rope sections are not [11, 11, 10, 0]");
    const Gguf::Kv *tok = g.get("tokenizer.ggml.tokens"), *pre = g.get("tokenizer.ggml.pre");
    if (!tok || tok->strs.size() != 248320) throw std::runtime_error("this GGUF's vocabulary is not Qwen3.5's 248320 tokens");
    if (!pre || pre->str != "qwen35") throw std::runtime_error("this GGUF's pre-tokenizer is " + (pre ? pre->str : std::string("missing")) + "; moex uses qwen35");
    // every tensor moex reads: present, the right shape, a type it can dequantize
    auto chk = [&](const std::string& n, std::vector<int64_t> ne) {
        const Gguf::Tensor& x = g.at(n);
        int bn, bb;
        if (!gg_block(x.type, bn, bb)) throw std::runtime_error("tensor " + n + " is " + gg_type_name(x.type) + "; moex imports " + GG_SUPPORTED);
        if (x.ne != ne) throw std::runtime_error("tensor " + n + " has an unexpected shape");
    };
    chk("token_embd.weight", {2048, 248320});
    chk(g.has("output.weight") ? "output.weight" : "token_embd.weight", {2048, 248320});
    chk("output_norm.weight", {2048});
    for (int l = 0; l < (int)bc; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        chk(p + "attn_norm.weight", {2048}); chk(p + "post_attention_norm.weight", {2048});
        chk(p + "ffn_gate_inp.weight", {2048, 256}); chk(p + "ffn_gate_inp_shexp.weight", {2048});
        chk(p + "ffn_gate_shexp.weight", {2048, 512}); chk(p + "ffn_up_shexp.weight", {2048, 512}); chk(p + "ffn_down_shexp.weight", {512, 2048});
        chk(p + "ffn_gate_exps.weight", {2048, 512, 256}); chk(p + "ffn_up_exps.weight", {2048, 512, 256}); chk(p + "ffn_down_exps.weight", {512, 2048, 256});
        if (l % 4 == 3 || l == 40) {
            chk(p + "attn_q.weight", {2048, 8192}); chk(p + "attn_k.weight", {2048, 512}); chk(p + "attn_v.weight", {2048, 512});
            chk(p + "attn_output.weight", {4096, 2048}); chk(p + "attn_q_norm.weight", {256}); chk(p + "attn_k_norm.weight", {256});
        } else {
            chk(p + "attn_qkv.weight", {2048, 8192}); chk(p + "attn_gate.weight", {2048, 4096}); chk(p + "ssm_alpha.weight", {2048, 32});
            chk(p + "ssm_beta.weight", {2048, 32}); chk(p + "ssm_out.weight", {4096, 2048}); chk(p + "ssm_a", {32});
            chk(p + "ssm_dt.bias", {32}); chk(p + "ssm_conv1d.weight", {4, 8192}); chk(p + "ssm_norm.weight", {128});
        }
    }
    if (mtp) {
        chk("blk.40.nextn.eh_proj.weight", {4096, 2048});
        for (auto n : {"enorm", "hnorm", "shared_head_norm"}) chk(std::string("blk.40.nextn.") + n + ".weight", {2048});
    }
}

// The dense tensors of block l under their container names, mapped back to the HF originals: RMSNorm weights are
// stored as 1 + w (all but the DeltaNet ssm_norm), DeltaNet value heads reordered, A_log stored as -exp(A_log).
static void gg_dense_layer(const Gguf& g, int l, const std::string& pre, std::vector<GgOut>& out) {
    const std::string p = "blk." + std::to_string(l) + ".";
    enum Fix { NONE, NORM, ROWS, QKV, COLS, ALOG, VEC, CONV };
    auto add = [&](const std::string& gname, const std::string& name, std::vector<int64_t> shape, Fix fix) {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        out.push_back({pre + name, "F16", shape, n * 2, [&g, gname, fix, shape](uint8_t* dst) {
            std::vector<float> w = g.all(gname);
            const int64_t cols = shape.back();
            if (fix == NORM) for (auto& v : w) v -= 1.0f;
            if (fix == ROWS) gg_unperm_rows(w, 0, (int64_t)shape[0] / GG_VH, cols);
            if (fix == QKV) gg_unperm_rows(w, 4096, GG_HD, cols);
            if (fix == CONV) gg_unperm_rows(w, 4096, GG_HD, 4);
            if (fix == VEC || fix == ALOG) gg_unperm_rows(w, 0, 1, 1);
            if (fix == ALOG) for (auto& v : w) v = std::log(-v);
            if (fix == COLS) {  // out_proj [2048, 4096]: columns grouped by value head
                std::vector<float> c(w);
                for (int64_t r = 0; r < shape[0]; r++)
                    for (int h = 0; h < GG_VH; h++) memcpy(&w[r * cols + h * GG_HD], &c[r * cols + gg_vhead_src(h) * GG_HD], GG_HD * 4);
            }
            for (size_t i = 0; i < w.size(); i++) ((uint16_t*)dst)[i] = gg_f2h(w[i]);
        }});
    };
    add(p + "attn_norm.weight", "input_layernorm.weight", {2048}, NORM);
    add(p + "post_attention_norm.weight", "post_attention_layernorm.weight", {2048}, NORM);
    add(p + "ffn_gate_inp.weight", "mlp.gate.weight", {256, 2048}, NONE);
    add(p + "ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", {1, 2048}, NONE);
    add(p + "ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", {512, 2048}, NONE);
    add(p + "ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", {512, 2048}, NONE);
    add(p + "ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", {2048, 512}, NONE);
    if (g.has(p + "attn_q.weight")) {
        add(p + "attn_q.weight", "self_attn.q_proj.weight", {8192, 2048}, NONE);
        add(p + "attn_k.weight", "self_attn.k_proj.weight", {512, 2048}, NONE);
        add(p + "attn_v.weight", "self_attn.v_proj.weight", {512, 2048}, NONE);
        add(p + "attn_output.weight", "self_attn.o_proj.weight", {2048, 4096}, NONE);
        add(p + "attn_q_norm.weight", "self_attn.q_norm.weight", {256}, NORM);
        add(p + "attn_k_norm.weight", "self_attn.k_norm.weight", {256}, NORM);
    } else {
        add(p + "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", {8192, 2048}, QKV);
        add(p + "attn_gate.weight", "linear_attn.in_proj_z.weight", {4096, 2048}, ROWS);
        add(p + "ssm_alpha.weight", "linear_attn.in_proj_a.weight", {32, 2048}, ROWS);
        add(p + "ssm_beta.weight", "linear_attn.in_proj_b.weight", {32, 2048}, ROWS);
        add(p + "ssm_out.weight", "linear_attn.out_proj.weight", {2048, 4096}, COLS);
        add(p + "ssm_a", "linear_attn.A_log", {32}, ALOG);
        add(p + "ssm_dt.bias", "linear_attn.dt_bias", {32}, VEC);
        add(p + "ssm_conv1d.weight", "linear_attn.conv1d.weight", {8192, 1, 4}, CONV);
        add(p + "ssm_norm.weight", "linear_attn.norm.weight", {128}, NONE);
    }
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.name < b.name; });
}

// Block l's 256 experts -> moex zero-point int4: per expert [gate IxH | up IxH | down HxI] packed nibbles and half2
// (s, m) per 32 inputs. Returns the largest per-expert relative RMS error against the GGUF's own values (Q4_K: only
// the fp16 rounding of s and m), for the log.
static double gg_experts(const Gguf& g, int l, std::vector<uint8_t>& q, std::vector<uint16_t>& s) {
    const std::string p = "blk." + std::to_string(l) + ".";
    const Gguf::Tensor* t[3] = {&g.at(p + "ffn_gate_exps.weight"), &g.at(p + "ffn_up_exps.weight"), &g.at(p + "ffn_down_exps.weight")};
    q.resize(GG_E * GG_EXPERT_Q); s.resize(GG_E * GG_EXPERT_S / 2);
    std::vector<double> err(GG_E);
    gg_par_for(GG_E, [&](int64_t e) {
        std::vector<float> w(GG_D), tmp(256);
        uint8_t* qe = q.data() + e * GG_EXPERT_Q;
        uint16_t* se = s.data() + e * GG_EXPERT_S / 2;
        double ee = 0, ss = 0;
        for (int part = 0; part < 3; part++) {
            const int rows = part < 2 ? GG_I : GG_D, cols = part < 2 ? GG_D : GG_I;
            for (int r = 0; r < rows; r++) {
                const int64_t row = (int64_t)part * GG_I + r;  // row of the merged [gate | up | down] layout
                uint8_t* qr = part < 2 ? qe + (size_t)row * (GG_D / 2) : qe + (size_t)GG_I * GG_D + (size_t)r * (GG_I / 2);
                uint16_t* sr = se + (part < 2 ? row * (GG_D / 32) : (size_t)2 * GG_I * (GG_D / 32) + (size_t)r * (GG_I / 32)) * 2;
                gg_zp_row(t[part]->type, t[part]->data + (e * rows + r) * t[part]->row_bytes, cols, qr, sr, tmp.data());
                g.rows(*t[part], e * rows + r, 1, w.data());
                for (int i = 0; i < cols; i++) {
                    const float v = gg_half(sr[i / 32 * 2]) * ((qr[i / 2] >> (i % 2 * 4)) & 0xF) + gg_half(sr[i / 32 * 2 + 1]);
                    ee += (double)(w[i] - v) * (w[i] - v); ss += (double)w[i] * w[i];
                }
            }
        }
        err[e] = ss > 0 ? std::sqrt(ee / ss) : 0;
    });
    return *std::max_element(err.begin(), err.end());
}

// One layer's (or the MTP layer's) safetensors file, laid out like the downloaded container: scales, dense, packed
static void gg_write_layer(const Gguf& g, int l, const std::string& pre, const std::string& path, const std::vector<GgOut>& extra, double& err) {
    std::vector<uint8_t> q; std::vector<uint16_t> s;
    err = gg_experts(g, l, q, s);
    std::vector<GgOut> ts, dense(extra);
    for (int e = 0; e < GG_E; e++)
        ts.push_back({pre + "mlp.experts." + std::to_string(e) + ".qs", "F16", {(int64_t)GG_EXPERT_S / 2}, GG_EXPERT_S,
                      [&s, e](uint8_t* d) { memcpy(d, (const uint8_t*)s.data() + e * GG_EXPERT_S, GG_EXPERT_S); }});
    gg_dense_layer(g, l, pre, dense);
    ts.insert(ts.end(), dense.begin(), dense.end());
    for (int e = 0; e < GG_E; e++)
        ts.push_back({pre + "mlp.experts." + std::to_string(e) + ".merged_weight", "U8", {(int64_t)GG_EXPERT_Q}, GG_EXPERT_Q,
                      [&q, e](uint8_t* d) { memcpy(d, q.data() + e * GG_EXPERT_Q, GG_EXPERT_Q); }});
    gg_write_st(path, ts);
}

static GgOut gg_f16_tensor(const Gguf& g, const std::string& gname, const std::string& name, std::vector<int64_t> shape, bool norm) {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return {name, "F16", shape, n * 2, [&g, gname, norm, n](uint8_t* dst) {
        std::vector<float> w = g.all(gname);
        gg_par_for((int64_t)(n + (1 << 20) - 1) >> 20, [&](int64_t c) {
            for (uint64_t i = (uint64_t)c << 20; i < std::min(n, (uint64_t)(c + 1) << 20); i++) ((uint16_t*)dst)[i] = gg_f2h(norm ? w[i] - 1.0f : w[i]);
        });
    }};
}

// `moex import FILE.gguf [--out DIR]`
static int import_gguf(const std::string& src, std::string out) {
    namespace fs = std::filesystem;
    if (out.empty()) out = (fs::path(src).parent_path() / (fs::path(src).stem().string() + "-moex")).string();
    const std::string part = out + ".part";
    try {
        if (fs::exists(fs::path(out) / "config.json")) {
            fprintf(stderr, "%s already holds a model. Use it: moex chat --model \"%s\" (or delete the folder to import again)\n", out.c_str(), out.c_str());
            return 1;
        }
        Gguf g(src);
        bool mtp = false;
        gg_check_shape(g, mtp);
        const Gguf::Kv* name = g.get("general.name");
        const uint64_t need = 40ull * (GG_E * (GG_EXPERT_Q + GG_EXPERT_S) + 80000000ull) + 2ull * 248320 * 2048 * 2 +
                              (mtp ? GG_E * (GG_EXPERT_Q + GG_EXPERT_S) + 80000000ull : 0);
        fs::create_directories(fs::path(out).parent_path().empty() ? fs::path(".") : fs::path(out).parent_path());
        const auto sp = fs::space(fs::path(out).parent_path().empty() ? fs::path(".") : fs::path(out).parent_path());
        fprintf(stderr, "Importing %s (%s)\n  into %s\n  needs %.1f GB of disk (%.1f GB free); the GGUF itself is not changed or needed afterwards.\n",
                src.c_str(), name ? name->str.c_str() : "qwen35moe", out.c_str(), need / 1e9, sp.available / 1e9);
        if (sp.available < need + (1ull << 30)) { fprintf(stderr, "Not enough free disk space.\n"); return 1; }
        fs::remove_all(part);
        fs::create_directories(part);
        const auto t0 = std::chrono::steady_clock::now();
        double worst = 0;
        for (int l = 0; l < 40; l++) {
            char fn[64]; snprintf(fn, sizeof fn, "/model-%05d.safetensors", l);
            double e;
            gg_write_layer(g, l, "model.layers." + std::to_string(l) + ".", part + fn, {}, e);
            worst = std::max(worst, e);
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "\r  layer %2d/40 (experts requantized to int4, worst rel. error %.3f)  %.0f s, ~%.0f s left   ",
                    l + 1, e, sec, sec / (l + 1) * (39 - l));
        }
        fprintf(stderr, "\n  embeddings and output head\n");
        gg_write_st(part + "/model-globals.safetensors", {
            gg_f16_tensor(g, g.has("output.weight") ? "output.weight" : "token_embd.weight", "lm_head.weight", {248320, 2048}, false),
            gg_f16_tensor(g, "token_embd.weight", "model.embed_tokens.weight", {248320, 2048}, false),
            gg_f16_tensor(g, "output_norm.weight", "model.norm.weight", {2048}, true)});
        if (mtp) {
            fprintf(stderr, "  MTP head (multi-token prediction)\n");
            double e;
            gg_write_layer(g, 40, "mtp.layers.0.", part + "/mtp.safetensors", {
                gg_f16_tensor(g, "blk.40.nextn.eh_proj.weight", "mtp.fc.weight", {2048, 4096}, false),
                gg_f16_tensor(g, "blk.40.nextn.shared_head_norm.weight", "mtp.norm.weight", {2048}, true),
                gg_f16_tensor(g, "blk.40.nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", {2048}, true),
                gg_f16_tensor(g, "blk.40.nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", {2048}, true)}, e);
            worst = std::max(worst, e);
        }
        auto put = [&](const std::string& fn, const std::string& text) {
            FILE* f = fopen((part + "/" + fn).c_str(), "wb");
            if (!f || fwrite(text.data(), 1, text.size(), f) != text.size() || fclose(f)) throw std::runtime_error("cannot write " + fn);
        };
        put("tokenizer.json", gg_tokenizer_json(g));
        if (const Gguf::Kv* ct = g.get("tokenizer.chat_template")) put("chat_template.jinja", ct->str);
        std::string cfg(GG_CONFIG_JSON), srcname = jesc(fs::path(src).filename().string());
        cfg.replace(cfg.find("%s"), 2, srcname);
        put("config.json", cfg);
        fs::rename(part, out);
        fprintf(stderr, "Done in %.0f s (experts: worst relative RMS error of the int4 requantization %.3f).\n%s"
                        "Run it: moex chat --model \"%s\"\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), worst,
                mtp ? "" : "This GGUF has no MTP layer: moex runs it without multi-token prediction.\n", out.c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "\nmoex import: %s\n", e.what());
        std::error_code ec;
        fs::remove_all(part, ec);
        return 1;
    }
}
