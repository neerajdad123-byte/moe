"""Fetch the MTP head of Qwen/Qwen3.6-35B-A3B (~1.7 GB of the 72 GB repo, HTTP range requests on the last two
shards) and convert it to the moex container format next to the model: mtp.safetensors.

  dense tensors (fc, norms, attention, shared expert, router)  -> F16, same names (mtp.*)
  experts (fused gate_up_proj / down_proj, BF16)               -> per expert, like the main layers:
      mtp.layers.0.mlp.experts.{e}.merged_weight  U8   int4 [gate IxH | up IxH | down HxI], low nibble first
      mtp.layers.0.mlp.experts.{e}.qs             F32  one scale per 64 inputs, symmetric (q in [-8, 7])

  python tools/fetch_mtp.py C:/models/qwen36_i4_gs64
Raw downloads are cached in <model>/mtp_raw/ (resumable); delete it after a successful conversion.
"""
import json, os, struct, sys, urllib.request
import numpy as np

REPO = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/"
SHARDS = ["model-00025-of-00026.safetensors", "model-00026-of-00026.safetensors"]
H, I, E, GS = 2048, 512, 256, 64


def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "moex"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != b - a + 1:
        raise IOError(f"range {a}-{b}: got {len(data)} bytes")
    return data


def fetch_tensor(url, base, name, info, raw_dir):
    path = os.path.join(raw_dir, name + ".bin")
    a, b = info["data_offsets"]
    want = b - a
    have = os.path.getsize(path) if os.path.exists(path) else 0
    with open(path, "ab") as f:
        while have < want:
            n = min(64 << 20, want - have)
            f.write(get(url, base + a + have, base + a + have + n - 1))
            have += n
            print(f"\r  {name:58s} {have / 2**20:8.0f} / {want / 2**20:.0f} MB", end="", flush=True)
    print()
    return np.fromfile(path, dtype=np.uint8)


def bf16(raw, shape):
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def q4_rows(w):
    """[rows, cols] fp32 -> (int4 bytes, f32 scales): symmetric per 64 inputs, round to nearest."""
    g = w.reshape(w.shape[0], -1, GS)
    s = np.abs(g).max(-1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    q = np.clip(np.rint(g / s), -8, 7).astype(np.int8).reshape(-1)
    u = (q & 0xF).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).astype(np.uint8), s.reshape(-1).astype(np.float32)


def write_safetensors(path, tensors):
    header, off = {}, 0
    for name, arr in tensors.items():
        dt = {np.float16: "F16", np.float32: "F32", np.uint8: "U8"}[arr.dtype.type]
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + arr.nbytes]}
        off += arr.nbytes
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for arr in tensors.values(): f.write(np.ascontiguousarray(arr).tobytes())
    os.replace(path + ".part", path)


def main(model_dir):
    raw_dir = os.path.join(model_dir, "mtp_raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw, shapes = {}, {}
    for shard in SHARDS:
        url = REPO + shard
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        hdr = json.loads(get(url, 8, 8 + n - 1))
        for name, info in hdr.items():
            if not name.startswith("mtp."): continue
            assert info["dtype"] == "BF16", (name, info["dtype"])
            raw[name] = fetch_tensor(url, 8 + n, name, info, raw_dir)
            shapes[name] = info["shape"]
    print("tensors:", {k: v for k, v in shapes.items()})
    out = {}
    for name, r in raw.items():
        if ".mlp.experts." in name: continue
        out[name] = bf16(r, shapes[name]).astype(np.float16)
    gu_name, dn_name = "mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"
    gu, dn = bf16(raw[gu_name], shapes[gu_name]), bf16(raw[dn_name], shapes[dn_name])
    # fused HF layout: gate_up [E, 2I, H] (gate rows first) and down [E, H, I]; transposed variants are flipped
    if gu.shape == (E, H, 2 * I): gu = gu.transpose(0, 2, 1)
    if dn.shape == (E, I, H): dn = dn.transpose(0, 2, 1)
    assert gu.shape == (E, 2 * I, H) and dn.shape == (E, H, I), (gu.shape, dn.shape)
    err = 0.0
    for e in range(E):
        parts = [q4_rows(gu[e, :I]), q4_rows(gu[e, I:]), q4_rows(dn[e])]
        p = f"mtp.layers.0.mlp.experts.{e}."
        out[p + "merged_weight"] = np.concatenate([q for q, _ in parts])
        out[p + "qs"] = np.concatenate([s for _, s in parts])
        if e == 0:  # self-check: dequantize the gate of expert 0 the way the engine does (ref/dequant_expert)
            q = out[p + "merged_weight"][: I * H // 2]
            v = np.stack([q & 0xF, q >> 4], -1).reshape(-1).astype(np.int16)
            v = np.where(v >= 8, v - 16, v).astype(np.float32).reshape(I, H // GS, GS)
            deq = (v * out[p + "qs"][: I * H // GS].reshape(I, -1, 1)).reshape(I, H)
            err = np.abs(deq - gu[0, :I]).max() / np.abs(gu[0, :I]).max()
    assert out["mtp.layers.0.mlp.experts.0.merged_weight"].size * 2 == 3 * I * H
    print(f"int4 self-check: expert 0 gate max abs error {err:.4f} of its largest weight (half a step is ~0.07)")
    assert err < 0.08
    dst = os.path.join(model_dir, "mtp.safetensors")
    write_safetensors(dst, out)
    print(f"wrote {dst}: {len(out)} tensors, {os.path.getsize(dst) / 1e9:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "C:/models/qwen36_i4_gs64")
