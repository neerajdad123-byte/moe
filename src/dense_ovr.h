// Packed dense matrices written by tools/quant_dense.py (MOEX_DENSE, research), decoded on the host. Host only
// (tests/test_dense.cpp). Per container tensor N:
//   N.s   f32 [rows][G]   group scales (group size I / G: 32 or 64)
//   N.q4  u8  [rows][I/2] 4-bit codes, low nibble = even column            (or)
//   N.q6  u8  [rows][I/2] + N.q6h u8 [rows][I/4]: 6-bit codes, low 4 bits as q4, high 2 bits 4 per byte
//   N.z   f32 [rows][G]   optional zero points (asymmetric: codes unsigned, w = (u - z) * s); without it codes are
//                         symmetric: q4 two's complement nibbles, q6 u - 32
//   N.isc f32 [I]         optional AWQ input-channel scales: the codes encode W * isc, so w = value / isc[i]
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "st.h"

// Calls put(k, w) for k = 0 .. rows * I - 1 (row-major).
template <class Put>
void dequant_packed(Container& C, const std::string& n, int64_t rows, int64_t I, Put put) {
    const bool six = C.has(n + ".q6");
    const uint8_t* lo = C.map(n + (six ? ".q6" : ".q4"));
    const uint8_t* hi = six ? C.map(n + ".q6h") : nullptr;
    const float* s = (const float*)C.map(n + ".s");
    const int64_t G = (int64_t)C.at(n + ".s").bytes / 4 / rows;
    if (G <= 0 || I % G || G * rows * 4 != (int64_t)C.at(n + ".s").bytes || (int64_t)C.at(n + (six ? ".q6" : ".q4")).bytes != rows * I / 2) {
        fprintf(stderr, "shape mismatch %s (packed)\n", n.c_str()); exit(1);
    }
    const int64_t gs = I / G;
    const float* z = C.has(n + ".z") ? (const float*)C.map(n + ".z") : nullptr;
    const float* isc = C.has(n + ".isc") ? (const float*)C.map(n + ".isc") : nullptr;
    for (int64_t k = 0; k < rows * I; k++) {
        int u = (lo[k / 2] >> (4 * (k & 1))) & 15;
        if (six) u |= ((hi[k / 4] >> (2 * (k & 3))) & 3) << 4;
        const int64_t r = k / I, i = k % I, g = r * G + i / gs;
        const float q = z ? (float)u - z[g] : (float)(six ? u - 32 : u - ((u & 8) << 1));
        put(k, isc ? q * s[g] / isc[i] : q * s[g]);
    }
}
