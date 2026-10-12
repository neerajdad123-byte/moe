"""Does llama.cpp's converter transform Qwen3.6 tensors? Compare GGUF tensors (dequantized) with the HF F16
originals in the colibri container: row order (head reordering), +1 norm offsets, A_log transforms.

  python tools/gguf_map_check.py C:/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
"""
import json, os, struct, sys
import numpy as np

GG = sys.argv[1]
CT = r"C:\models\qwen36_i4_gs64"

def gguf_index(path):
    f = open(path, "rb"); data = f.read(64 << 20)
    pos = 4
    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos); pos += struct.calcsize("<" + fmt); return v[0]
    def rstr():
        nonlocal pos
        n = rd("Q"); s = data[pos:pos + n].decode("utf-8", "replace"); pos += n; return s
    SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            at, n = rd("I"), rd("Q"); return [rval(at) for _ in range(n)]
        return rd(SC[t])
    ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
    kv = {}
    for _ in range(nkv):
        k = rstr(); kv[k] = rval(rd("I"))
    tens = {}
    for _ in range(nt):
        name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); off = rd("Q")
        tens[name] = (dims, t, off)
    align = kv.get("general.alignment", 32)
    start = (pos + align - 1) // align * align
    return tens, start

tens, start = gguf_index(GG)
mm = np.memmap(GG, dtype=np.uint8, mode="r")

def gg(name):
    dims, t, off = tens[name]
    n = int(np.prod(dims)); base = start + off
    if t == 0:  # F32
        return np.frombuffer(mm[base:base + 4 * n], np.float32).reshape(dims[::-1])
    if t == 8:  # Q8_0: blocks of 32 = fp16 d + 32 int8
        nb = n // 32
        blk = np.frombuffer(mm[base:base + 34 * nb], np.uint8).reshape(nb, 34)
        d = blk[:, :2].copy().view(np.float16).astype(np.float32)
        q = blk[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(dims[::-1])
    raise ValueError(f"type {t} not handled here")

def ct_index(root):
    idx = {}
    for fn in os.listdir(root):
        if fn.endswith(".safetensors"):
            p = os.path.join(root, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
            for k, v in h.items():
                if k != "__metadata__": idx[k] = (p, 8 + n, v)
    return idx
cidx = ct_index(CT)
def ct(name):
    p, base, v = cidx[name]; a, b = v["data_offsets"]
    dt = {"F16": np.float16, "F32": np.float32}[v["dtype"]]
    return np.fromfile(p, dtype=np.uint8, count=b - a, offset=base + a).view(dt).astype(np.float32).reshape(v["shape"])

def rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))

def rows_match(G, H, label):
    """Is G a row permutation of H? Report identity error and the best-match permutation if not identity."""
    e_id = rel(G, H)
    if e_id < 0.02:
        print(f"  {label:28s} same order, rel err {e_id:.3e} (Q8_0 noise)"); return None
    # find for each GGUF row the closest HF row (cosine)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12); Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    perm = np.argmax(Gn @ Hn.T, axis=1)
    e_p = rel(G, H[perm])
    print(f"  {label:28s} NOT same order (rel {e_id:.2f}); best row permutation gives rel {e_p:.3e}; first map: {perm[:16].tolist()}")
    return perm

L = 0
print("layer 0 (DeltaNet):")
pq = rows_match(gg(f"blk.{L}.attn_qkv.weight"), ct(f"model.layers.{L}.linear_attn.in_proj_qkv.weight"), "attn_qkv vs in_proj_qkv")
pz = rows_match(gg(f"blk.{L}.attn_gate.weight"), ct(f"model.layers.{L}.linear_attn.in_proj_z.weight"), "attn_gate vs in_proj_z")
pa = rows_match(gg(f"blk.{L}.ssm_alpha.weight"), ct(f"model.layers.{L}.linear_attn.in_proj_a.weight"), "ssm_alpha vs in_proj_a")
pb = rows_match(gg(f"blk.{L}.ssm_beta.weight"), ct(f"model.layers.{L}.linear_attn.in_proj_b.weight"), "ssm_beta vs in_proj_b")
G = gg(f"blk.{L}.ssm_out.weight"); H = ct(f"model.layers.{L}.linear_attn.out_proj.weight")
po = rows_match(G.T, H.T, "ssm_out (columns) vs out_proj")
A = ct(f"model.layers.{L}.linear_attn.A_log"); sa = gg(f"blk.{L}.ssm_a")
print(f"  ssm_a vs -exp(A_log): rel {rel(sa, -np.exp(A)):.3e} | vs A_log: rel {rel(sa, A):.3e}")
print(f"  ssm_dt.bias vs dt_bias: rel {rel(gg(f'blk.{L}.ssm_dt.bias'), ct(f'model.layers.{L}.linear_attn.dt_bias')):.3e}")
cw = gg(f"blk.{L}.ssm_conv1d.weight"); hc = ct(f"model.layers.{L}.linear_attn.conv1d.weight").reshape(8192, 4)
print(f"  ssm_conv1d vs conv1d: rel {rel(cw, hc):.3e}")
print(f"  ssm_norm vs norm.weight: rel {rel(gg(f'blk.{L}.ssm_norm.weight'), ct(f'model.layers.{L}.linear_attn.norm.weight')):.3e}")
for g_name, c_name in (("attn_norm", "input_layernorm"), ("post_attention_norm", "post_attention_layernorm")):
    gv = gg(f"blk.{L}.{g_name}.weight"); cv = ct(f"model.layers.{L}.{c_name}.weight")
    print(f"  {g_name} vs {c_name}: rel {rel(gv, cv):.3e} | vs 1+w: rel {rel(gv, 1 + cv):.3e}")
print(f"  ffn_gate_inp vs gate (router): rel {rel(gg(f'blk.{L}.ffn_gate_inp.weight'), ct(f'model.layers.{L}.mlp.gate.weight')):.3e}")
print(f"  ffn_gate_inp_shexp vs shared_expert_gate: rel {rel(gg(f'blk.{L}.ffn_gate_inp_shexp.weight').reshape(1, -1), ct(f'model.layers.{L}.mlp.shared_expert_gate.weight')):.3e}")
rows_match(gg(f"blk.{L}.ffn_gate_shexp.weight"), ct(f"model.layers.{L}.mlp.shared_expert.gate_proj.weight"), "ffn_gate_shexp")
rows_match(gg(f"blk.{L}.ffn_down_shexp.weight"), ct(f"model.layers.{L}.mlp.shared_expert.down_proj.weight"), "ffn_down_shexp")
L = 3
print("layer 3 (attention):")
rows_match(gg(f"blk.{L}.attn_q.weight"), ct(f"model.layers.{L}.self_attn.q_proj.weight"), "attn_q vs q_proj")
rows_match(gg(f"blk.{L}.attn_k.weight"), ct(f"model.layers.{L}.self_attn.k_proj.weight"), "attn_k vs k_proj")
rows_match(gg(f"blk.{L}.attn_v.weight"), ct(f"model.layers.{L}.self_attn.v_proj.weight"), "attn_v vs v_proj")
rows_match(gg(f"blk.{L}.attn_output.weight").T, ct(f"model.layers.{L}.self_attn.o_proj.weight").T, "attn_output (cols)")
for g_name, c_name in (("attn_q_norm", "self_attn.q_norm"), ("attn_k_norm", "self_attn.k_norm")):
    gv = gg(f"blk.{L}.{g_name}.weight"); cv = ct(f"model.layers.{L}.{c_name}.weight")
    print(f"  {g_name}: rel {rel(gv, cv):.3e} | vs 1+w: rel {rel(gv, 1 + cv):.3e}")
gv = gg("output_norm.weight"); cv = ct("model.norm.weight")
print(f"output_norm vs model.norm: rel {rel(gv, cv):.3e} | vs 1+w: rel {rel(gv, 1 + cv):.3e}")

# ---- verify the value-head permutation: GGUF head j holds HF head 2*(j % 16) + j // 16
print("value-head un-permutation check (layer 0):")
L = 0
perm = np.array([2 * (j % 16) + j // 16 for j in range(32)])  # GGUF head -> HF head
inv = np.argsort(perm)                                         # HF head -> GGUF head
def unperm_rows(G, head):  # G rows grouped by GGUF value head (head rows each) -> HF order
    return G.reshape(32, head, -1)[inv].reshape(G.shape)
z = unperm_rows(gg(f"blk.{L}.attn_gate.weight"), 128)
print(f"  attn_gate unpermuted vs in_proj_z: rel {rel(z, ct(f'model.layers.{L}.linear_attn.in_proj_z.weight')):.3e}")
for g_name, c_name in (("ssm_alpha", "in_proj_a"), ("ssm_beta", "in_proj_b")):
    print(f"  {g_name} unpermuted vs {c_name}: rel {rel(unperm_rows(gg(f'blk.{L}.{g_name}.weight'), 1), ct(f'model.layers.{L}.linear_attn.{c_name}.weight')):.3e}")
qkv = gg(f"blk.{L}.attn_qkv.weight"); qkv_u = np.concatenate([qkv[:4096], unperm_rows(qkv[4096:], 128)])
print(f"  attn_qkv (v part unpermuted) vs in_proj_qkv: rel {rel(qkv_u, ct(f'model.layers.{L}.linear_attn.in_proj_qkv.weight')):.3e}")
so = gg(f"blk.{L}.ssm_out.weight"); so_u = unperm_rows(so.T, 128).T
print(f"  ssm_out (cols unpermuted) vs out_proj: rel {rel(so_u, ct(f'model.layers.{L}.linear_attn.out_proj.weight')):.3e}")
sa = gg(f"blk.{L}.ssm_a")[inv]; A = ct(f"model.layers.{L}.linear_attn.A_log")
print(f"  ssm_a unpermuted vs -exp(A_log): rel {rel(sa, -np.exp(A)):.3e}")
print(f"  ssm_dt.bias unpermuted vs dt_bias: rel {rel(gg(f'blk.{L}.ssm_dt.bias')[inv], ct(f'model.layers.{L}.linear_attn.dt_bias')):.3e}")
cw = gg(f"blk.{L}.ssm_conv1d.weight"); cw_u = np.concatenate([cw[:4096], unperm_rows(cw[4096:], 128)])
print(f"  ssm_conv1d (v channels unpermuted) vs conv1d: rel {rel(cw_u, ct(f'model.layers.{L}.linear_attn.conv1d.weight').reshape(8192, 4)):.3e}")
