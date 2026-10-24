"""Compare a `moex import`ed container with the reference container (same base model): every dense F16 tensor
(max abs error relative to the tensor's max), a sample of experts (int4 dequantized, relative RMS), the tokenizer
(vocab, merges, added tokens) and the tensor name sets. Mapping mistakes show up as errors near 1, quantization
noise as ~1e-3 .. 1e-2.

  python tools/gguf_import_check.py C:/models/gguf/Qwen3.6-35B-A3B-Q4_K_M-moex [C:/models/qwen36_i4_gs64]
"""
import json, os, struct, sys
import numpy as np

IMP = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else r"C:\models\qwen36_i4_gs64"


def index(root):
    idx = {}
    for fn in sorted(os.listdir(root)):
        if fn.endswith(".safetensors"):
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
            for k, v in h.items():
                if k != "__metadata__": idx[k] = (p, 8 + n, v)
    return idx


def load(idx, name):
    p, base, v = idx[name]; a, b = v["data_offsets"]
    dt = {"F16": np.float16, "F32": np.float32, "U8": np.uint8}[v["dtype"]]
    return np.fromfile(p, dtype=np.uint8, count=b - a, offset=base + a).view(dt).reshape(v["shape"])


def expert(idx, prefix):  # ref/qwen36_ref.py dequant_expert: [gate IxH | up IxH | down HxI]
    q = load(idx, prefix + "merged_weight"); s = load(idx, prefix + "qs")
    v = np.stack([q & 0xF, q >> 4], -1).reshape(-1).astype(np.int16)
    if s.dtype == np.uint8:  # zero-point experts: w = d * sc * u + dm * mn, [sc, mn] per 32, (d, dm) fp16 per 256
        nch = v.size // 32
        cs = s[:2 * nch].reshape(-1, 2)
        dd = s[2 * nch:].view(np.float16).astype(np.float32).reshape(-1, 1, 2).repeat(8, 1).reshape(-1, 2)
        sc, mn = dd[:, 0] * cs[:, 0], dd[:, 1] * cs[:, 1].view(np.int8)
        return (v.astype(np.float32).reshape(-1, 32) * sc[:, None] + mn[:, None]).reshape(-1)
    v = np.where(v >= 8, v - 16, v).astype(np.float32)
    return (v.reshape(-1, 64) * s.reshape(-1, 1)).reshape(-1)


a, b = index(IMP), index(REF)
only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
if not any(n.startswith("mtp.") for n in a):  # a GGUF without the MTP layer: no mtp.safetensors (moex runs without MTP)
    only_b = [n for n in only_b if not n.startswith("mtp.")]
print(f"tensors: imported {len(a)}, reference {len(b)}; only imported {only_a[:5]}, only reference {only_b[:5]}")
worst = {}
for name in sorted(set(a) & set(b)):
    if ".experts." in name: continue
    x, y = load(a, name).astype(np.float32), load(b, name).astype(np.float32)
    if x.shape != y.shape:
        print(f"  SHAPE {name}: {x.shape} vs {y.shape}"); worst[name] = 9.0; continue
    worst[name] = float(np.abs(x - y).max() / (np.abs(y).max() + 1e-30))
kinds = {}
for name, e in worst.items():
    k = ".".join(p for p in name.split(".") if not p.isdigit())
    kinds.setdefault(k, []).append((e, name))
print("dense tensors (max abs error / max abs, worst layer):")
for k in sorted(kinds):
    e, n = max(kinds[k])
    print(f"  {k:58s} {e:.2e}  {'<-- MISMATCH' if e > 0.2 else ''}  ({n})")
rng = np.random.default_rng(0)
errs = []
for l in (0, 3, 17, 39):
    for e in rng.integers(0, 256, 4):
        p = f"model.layers.{l}.mlp.experts.{e}."
        x, y = expert(a, p), expert(b, p)
        errs.append(np.sqrt(((x - y) ** 2).mean() / (y ** 2).mean()))
if "mtp.layers.0.mlp.experts.0.qs" in a and "mtp.layers.0.mlp.experts.0.qs" in b:
    x, y = expert(a, "mtp.layers.0.mlp.experts.0."), expert(b, "mtp.layers.0.mlp.experts.0.")
    errs.append(np.sqrt(((x - y) ** 2).mean() / (y ** 2).mean()))
print(f"experts (int4 vs int4, relative RMS over {len(errs)} experts): mean {np.mean(errs):.3f}, max {np.max(errs):.3f}")
ta = json.load(open(os.path.join(IMP, "tokenizer.json"), encoding="utf-8"))
tb = json.load(open(os.path.join(REF, "tokenizer.json"), encoding="utf-8"))
print("tokenizer: vocab identical", ta["model"]["vocab"] == tb["model"]["vocab"], "| merges identical", ta["model"]["merges"] == tb["model"]["merges"],
      "| pre_tokenizer identical", ta["pre_tokenizer"] == tb["pre_tokenizer"])
ad = {t["id"]: (t["content"], t["special"]) for t in ta["added_tokens"]}
bd = {t["id"]: (t["content"], t["special"]) for t in tb["added_tokens"]}
print("  added tokens differing:", {i: (ad.get(i), bd.get(i)) for i in sorted(set(ad) | set(bd)) if ad.get(i) != bd.get(i)})
bad = [n for n, e in worst.items() if e > 0.2]
print("MISMATCHES:" if bad else "all dense tensors match to quantization noise", bad[:10])
sys.exit(1 if bad or only_b else 0)
