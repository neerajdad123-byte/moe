"""Make tests/gguf_blocks.bin: GGUF quant blocks and their dequantization by the `gguf` Python package (llama.cpp's
reference), for tests/test_gguf.cpp. Real blocks from a GGUF file where it has the type, random blocks otherwise.

  python tools/make_gguf_fixture.py C:/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf

Record: u32 ggml type, u32 element count, the raw blocks, then the expected f32 values.
"""
import struct, sys
import numpy as np
import gguf
from gguf.quants import dequantize

T = gguf.GGMLQuantizationType
rng = np.random.default_rng(1)
r = gguf.GGUFReader(sys.argv[1])
real = {}
for t in r.tensors:
    if t.tensor_type not in real:
        real[t.tensor_type] = np.asarray(t.data).view(np.uint8).reshape(-1)

out = bytearray()
# byte offsets of the fp16 scales that random blocks get sane values at (others: random bytes are valid as they are)
FP16_AT = {T.Q4_0: [0], T.IQ4_NL: [0], T.Q5_0: [0], T.Q4_1: [0, 2], T.Q5_1: [0, 2], T.Q2_K: [80, 82], T.Q3_K: [108],
           T.IQ4_XS: [0], T.IQ3_XXS: [0], T.IQ3_S: [0]}
for qt in (T.F32, T.F16, T.BF16, T.Q8_0, T.Q4_K, T.Q5_K, T.Q6_K,
           T.Q4_0, T.Q4_1, T.Q5_0, T.Q5_1, T.Q2_K, T.Q3_K, T.IQ4_NL, T.IQ4_XS, T.IQ3_XXS, T.IQ3_S, T.MXFP4):
    bn, bb = gguf.GGML_QUANT_SIZES[qt]
    nb = 16 if bn > 1 else 512
    if qt in real:
        src = "real"
        raw = real[qt][: nb * bb].copy()
    else:
        src = "random"
        raw = rng.integers(0, 256, nb * bb, dtype=np.uint8)
        blk = raw.reshape(nb, bb)
        if qt == T.F16:
            blk[:] = rng.standard_normal(nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
        if qt == T.BF16:
            blk[:] = (rng.standard_normal(nb).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(nb, 2)
        if qt in (T.Q4_K, T.Q5_K):  # sane d, dmin (fp16) in front
            blk[:, 0:4] = np.stack([rng.uniform(1e-4, 1e-2, nb), rng.uniform(1e-4, 1e-2, nb)], 1).astype(np.float16).view(np.uint8)
        if qt == T.Q8_0:
            blk[:, 0:2] = rng.uniform(1e-4, 1e-2, (nb, 1)).astype(np.float16).view(np.uint8)
        if qt == T.Q6_K:
            blk[:, 208:210] = rng.uniform(1e-4, 1e-2, (nb, 1)).astype(np.float16).view(np.uint8)
        for o in FP16_AT.get(qt, []):
            blk[:, o:o + 2] = rng.uniform(1e-4, 1e-2, (nb, 1)).astype(np.float16).view(np.uint8)
        if qt == T.MXFP4:  # E8M0 exponent: scales around 2^-8
            blk[:, 0] = rng.integers(115, 125, nb)
    want = dequantize(raw, qt).astype(np.float32).reshape(-1)
    assert np.isfinite(want).all(), qt
    out += struct.pack("<II", int(qt), want.size) + raw.tobytes() + want.tobytes()
    print(f"{qt.name:5s} {src:6s} {want.size} values")
open("tests/gguf_blocks.bin", "wb").write(out)
print(f"wrote tests/gguf_blocks.bin ({len(out)} bytes)")
