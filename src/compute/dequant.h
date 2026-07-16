// MoEx clean-room K-quant dequantization.
//
// Implements the published GGML block formats (Q4_K, Q5_K, Q6_K, Q8_0) from
// their format description. Block byte layouts were confirmed against the
// public size invariants; no ggml source is used. All outputs are f32.
//
// Block layouts (QK_K = 256 elems/superblock):
//   Q8_0: 32 elems/block, 34 B  = fp16 d + 32 int8
//   Q4_K: 256/superblock, 144 B = fp16 d, fp16 dmin, 12 B packed 6-bit
//                                 scales/mins, 128 B of 4-bit quants
//   Q5_K: 256/superblock, 176 B = fp16 d, fp16 dmin, 12 B scales, 32 B high
//                                 bits, 128 B of 4-bit quants
//   Q6_K: 256/superblock, 210 B = 128 B low nibbles, 64 B high 2-bits,
//                                 16 int8 scales, fp16 d
#pragma once

#include <cstdint>

#include "gguf/gguf.h"

namespace moex {

// Decode a half-precision float (IEEE 754 binary16) to float.
float fp16_to_f32(uint16_t h);

// Dequantize `n` elements starting at `src` (raw tensor bytes of type `t`)
// into `dst` (f32). `n` must be a multiple of the type's block element count.
// Returns false for unsupported types or misaligned n.
bool dequantize_row(gguf::GgmlType t, const uint8_t* src, float* dst, uint64_t n);

}  // namespace moex
