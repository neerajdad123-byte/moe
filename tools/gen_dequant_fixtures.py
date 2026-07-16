#!/usr/bin/env python3
"""Generate golden dequant fixtures from the real blob using the pure-numpy
gguf reference as an EXTERNAL ORACLE (fixture generation only, not a MoEx
runtime dependency). For each of several tensors covering Q4_K/Q5_K/Q6_K/Q8_0,
dump the raw quantized bytes and the oracle f32 output so the clean-room C++
dequantizer can be validated bit-for-bit-of-intent against it.

Output: build/fixtures/<name>.raw   (quantized bytes as stored)
        build/fixtures/<name>.f32   (little-endian float32 oracle output)
        build/fixtures/manifest.txt (name type n raw_bytes)
"""
import os
import sys
import numpy as np
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType
import gguf.quants as q

BLOB = sys.argv[1] if len(sys.argv) > 1 else r"C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
OUT = os.path.join("build", "fixtures")
os.makedirs(OUT, exist_ok=True)

# Pick tensors that exercise each quant type present in the blob.
WANT = {
    "token_embd.weight": "Q4_K",
    "output.weight": "Q6_K",
    "blk.0.attn_output.weight": "Q5_K",
    "blk.0.attn_k.weight": "Q8_0",
    "blk.0.ffn_down_exps.weight": "Q6_K",   # expert stack (Q6_K)
    "blk.0.ffn_gate_exps.weight": "Q4_K",   # expert stack (Q4_K)
}

TYPE_MAP = {
    GGMLQuantizationType.Q4_K: "Q4_K",
    GGMLQuantizationType.Q5_K: "Q5_K",
    GGMLQuantizationType.Q6_K: "Q6_K",
    GGMLQuantizationType.Q8_0: "Q8_0",
    GGMLQuantizationType.F32: "F32",
}

reader = GGUFReader(BLOB)
by_name = {t.name: t for t in reader.tensors}

manifest = []
for name, want_type in WANT.items():
    if name not in by_name:
        print(f"  skip {name}: not found")
        continue
    t = by_name[name]
    tname = TYPE_MAP.get(t.tensor_type, str(t.tensor_type))
    # Cap the number of superblocks we test to keep fixtures small but
    # meaningful: 8 superblocks (2048 elems) for K-quants, 2048 for Q8_0.
    raw = t.data.tobytes()
    if tname in ("Q4_K", "Q5_K", "Q6_K"):
        blk_bytes = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}[tname]
        nblk = 8
        n_elems = nblk * 256
        raw = raw[: nblk * blk_bytes]
    elif tname == "Q8_0":
        nblk = 64
        n_elems = nblk * 32
        raw = raw[: nblk * 34]
    else:  # F32
        n_elems = 2048
        raw = raw[: n_elems * 4]

    # Oracle dequant: reshape raw into the block-count shape the ref expects.
    arr = np.frombuffer(raw, dtype=np.uint8)
    ref = q.dequantize(arr.reshape(1, -1), t.tensor_type).astype(np.float32).ravel()
    ref = ref[:n_elems]

    safe = name.replace(".", "_").replace("/", "_")
    with open(os.path.join(OUT, safe + ".raw"), "wb") as f:
        f.write(raw)
    with open(os.path.join(OUT, safe + ".f32"), "wb") as f:
        f.write(ref.tobytes())
    manifest.append(f"{safe} {tname} {n_elems} {len(raw)}")
    print(f"  {name:32s} {tname:5s} n={n_elems} raw={len(raw)}B ref[0:3]={ref[:3]}")

with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write("\n".join(manifest) + "\n")
print(f"wrote {len(manifest)} fixtures to {OUT}")
