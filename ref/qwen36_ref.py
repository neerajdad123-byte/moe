"""fp32 PyTorch reference for Qwen3.6-35B-A3B (colibri int4 gs64 container).

The ground truth every CUDA kernel is checked against. Token-by-token (decode-style)
forward only, so it mirrors exactly what the engine computes per step.
Dense F16 weights live on the GPU and are upcast per matmul; experts are read from
the mmapped container and dequantized on demand.

  python ref/qwen36_ref.py "The capital of France is" --n 16
"""
import argparse, json, os, struct, time
import numpy as np
import torch
import torch.nn.functional as F

MODEL = os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64")


class Container:
    """safetensors files read through np.memmap (no extra dependency)."""
    DT = {"F16": np.float16, "F32": np.float32, "U8": np.uint8}

    def __init__(self, root):
        self.index = {}
        for fn in sorted(os.listdir(root)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(root, fn)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.index[k] = (mm, 8 + n, v)

    def np(self, name):
        mm, base, v = self.index[name]
        a, b = v["data_offsets"]
        return mm[base + a:base + b].view(self.DT[v["dtype"]]).reshape(v["shape"])

    def t(self, name, dev):
        return torch.from_numpy(np.array(self.np(name))).to(dev)


def unpack_int4(b):
    """uint8 tensor -> fp32, low nibble = element 2k, high = 2k+1, signed [-8,7]."""
    q = torch.stack([b & 0xF, b >> 4], -1).reshape(-1).to(torch.int16)
    return torch.where(q >= 8, q - 16, q).float()


def dequant_expert(mw, qs, H, I, gs):
    """merged [gate IxH | up IxH | down HxI] int4 + f32 group scales -> 3 fp32 matrices."""
    q = unpack_int4(mw)
    n_gu, n_dn = I * H, H * I
    s_gu, s_dn = I * (H // gs), H * (I // gs)
    gate = (q[:n_gu].reshape(I, H // gs, gs) * qs[:s_gu].reshape(I, -1, 1)).reshape(I, H)
    up = (q[n_gu:2 * n_gu].reshape(I, H // gs, gs) * qs[s_gu:2 * s_gu].reshape(I, -1, 1)).reshape(I, H)
    down = (q[2 * n_gu:].reshape(H, I // gs, gs) * qs[2 * s_gu:].reshape(H, -1, 1)).reshape(H, I)
    return gate, up, down


def rmsnorm(x, w, eps, plus_one=True):
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * r * ((1.0 + w) if plus_one else w)


def rope_partial(x, pos, rot, theta):
    """rotate_half on the first `rot` dims of each head (x: [heads, head_dim])."""
    h = rot // 2
    j = torch.arange(h, device=x.device, dtype=torch.float32)
    ang = pos * theta ** (-2.0 * j / rot)
    cs, sn = torch.cos(ang), torch.sin(ang)
    a, b = x[:, :h].clone(), x[:, h:rot].clone()
    x[:, :h] = a * cs - b * sn
    x[:, h:rot] = b * cs + a * sn
    return x


class Qwen36Ref:
    def __init__(self, root=MODEL, dev="cuda"):
        self.dev = dev
        self.c = json.load(open(os.path.join(root, "qwen36_meta.json")))
        self.st = Container(root)
        c = self.c
        self.L = c["n_layers"]
        self.is_attn = [t == "full_attention" for t in c["layer_types"]]
        self.gs = c["expert_gs"]
        self.rot = int(c["head_dim"] * c["partial_rotary_factor"])
        g = lambda n: self.st.t(n, dev)
        self.embed = self.st.np("model.embed_tokens.weight")      # stays in RAM: one row per token
        self.lm_head = g("lm_head.weight")
        self.norm = g("model.norm.weight").float()
        self.W = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            names = ["input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight",
                     "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                     "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight"]
            if self.is_attn[i]:
                names += [f"self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
            else:
                names += ["linear_attn." + n for n in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                          "in_proj_b.weight", "out_proj.weight", "conv1d.weight", "norm.weight", "A_log", "dt_bias")]
            self.W.append({n: g(p + n) for n in names})
        self.reset()

    def reset(self):
        c, dev = self.c, self.dev
        self.pos = 0
        self.K = [[] for _ in range(self.L)]
        self.V = [[] for _ in range(self.L)]
        self.ring = [torch.zeros(c["dn_conv_dim"], c["dn_convk"] - 1, device=dev) for _ in range(self.L)]
        self.S = [torch.zeros(c["dn_vheads"], c["dn_kdim"], c["dn_vdim"], device=dev) for _ in range(self.L)]

    def expert(self, i, e):
        p = f"model.layers.{i}.mlp.experts.{e}."
        return dequant_expert(self.st.t(p + "merged_weight", self.dev), self.st.t(p + "qs", self.dev),
                              self.c["hidden"], self.c["moe_inter"], self.gs)

    @staticmethod
    def mv(w, x):
        return w.float() @ x

    def attention(self, i, x, dump):
        c, w = self.c, self.W[i]
        H, KV, hd = c["q_heads"], c["kv_heads"], c["head_dim"]
        q = self.mv(w["self_attn.q_proj.weight"], x).view(H, 2 * hd)
        query, gate = q[:, :hd].clone(), q[:, hd:]
        k = self.mv(w["self_attn.k_proj.weight"], x).view(KV, hd)
        v = self.mv(w["self_attn.v_proj.weight"], x).view(KV, hd)
        query = rope_partial(rmsnorm(query, w["self_attn.q_norm.weight"].float(), c["rms_eps"]), self.pos, self.rot, c["rope_theta"])
        k = rope_partial(rmsnorm(k, w["self_attn.k_norm.weight"].float(), c["rms_eps"]), self.pos, self.rot, c["rope_theta"])
        self.K[i].append(k); self.V[i].append(v)
        Kc, Vc = torch.stack(self.K[i], 1), torch.stack(self.V[i], 1)     # [KV, T, hd]
        Kr, Vr = Kc.repeat_interleave(H // KV, 0), Vc.repeat_interleave(H // KV, 0)
        sc = torch.einsum("hd,htd->ht", query, Kr) / hd ** 0.5
        ctx = torch.einsum("ht,htd->hd", torch.softmax(sc, -1), Vr)
        ag = (ctx * torch.sigmoid(gate)).reshape(-1)
        out = self.mv(w["self_attn.o_proj.weight"], ag)
        dump(f"attn_q", query); dump("attn_k", k); dump("attn_v", v); dump("attn_ctx", ctx); dump("attn_out", out)
        return out

    def deltanet(self, i, x, dump):
        c, w = self.c, self.W[i]
        vh, kh, kd, vd = c["dn_vheads"], c["dn_kheads"], c["dn_kdim"], c["dn_vdim"]
        p = "linear_attn."
        qkv = self.mv(w[p + "in_proj_qkv.weight"], x)
        z = self.mv(w[p + "in_proj_z.weight"], x).view(vh, vd)
        b = self.mv(w[p + "in_proj_b.weight"], x)
        a = self.mv(w[p + "in_proj_a.weight"], x)
        beta = torch.sigmoid(b)
        g = -torch.exp(w[p + "A_log"].float()) * F.softplus(a + w[p + "dt_bias"].float())
        cw = w[p + "conv1d.weight"].float().squeeze(1)                  # [C, convk]
        ring = self.ring[i]
        conv = F.silu((cw[:, :-1] * ring).sum(1) + cw[:, -1] * qkv)
        self.ring[i] = torch.cat([ring[:, 1:], qkv[:, None]], 1)
        q = conv[:kh * kd].view(kh, kd).repeat_interleave(vh // kh, 0)
        k = conv[kh * kd:2 * kh * kd].view(kh, kd).repeat_interleave(vh // kh, 0)
        v = conv[2 * kh * kd:].view(vh, vd)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / kd ** 0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        S = self.S[i] * torch.exp(g)[:, None, None]
        kv = torch.einsum("hk,hkv->hv", k, S)
        delta = (v - kv) * beta[:, None]
        S = S + torch.einsum("hk,hv->hkv", k, delta)
        self.S[i] = S
        o = torch.einsum("hk,hkv->hv", q, S)
        o = rmsnorm(o, w[p + "norm.weight"].float(), c["rms_eps"], plus_one=False) * F.silu(z)
        out = self.mv(w[p + "out_proj.weight"], o.reshape(-1))
        dump("dn_qkv", qkv); dump("dn_conv", conv); dump("dn_beta", beta); dump("dn_g", g)
        dump("dn_o", o); dump("dn_out", out)
        return out

    def moe(self, i, x, dump):
        c, w = self.c, self.W[i]
        logits = self.mv(w["mlp.gate.weight"], x)
        probs = torch.softmax(logits, -1)
        val, idx = torch.topk(probs, c["topk"])
        # Renormalize the top-k weights: the HF config omits norm_topk_prob and the Qwen3.5-MoE default is true
        # (colibri's meta says false; with false the model degenerates into loops, llama.cpp normalizes).
        if os.environ.get("REF_NO_NORM_TOPK") is None:
            val = val / val.sum()
        out = torch.zeros_like(x)
        for e, wt in zip(idx.tolist(), val.tolist()):
            gw, uw, dw = self.expert(i, e)
            out += wt * (dw @ (F.silu(gw @ x) * (uw @ x)))
        sh = self.mv(w["mlp.shared_expert.down_proj.weight"],
                     F.silu(self.mv(w["mlp.shared_expert.gate_proj.weight"], x)) * self.mv(w["mlp.shared_expert.up_proj.weight"], x))
        sg = torch.sigmoid(self.mv(w["mlp.shared_expert_gate.weight"], x))
        out += sg * sh
        dump("router_logits", logits); dump("route_idx", idx.int()); dump("route_w", val); dump("moe_out", out)
        self.routes.append(idx.tolist())
        return out

    @torch.no_grad()
    def step(self, tok, dump_dir=None):
        """One token through the model. Returns fp32 logits [vocab]."""
        def dump(name, t):
            if dump_dir and (not self.logits_only or name == "logits"):
                fn = f"p{self.pos:03d}_" + (f"L{cur:02d}_" if cur is not None else "") + name + ".bin"
                t.detach().cpu().contiguous().numpy().tofile(os.path.join(dump_dir, fn))
        cur = None
        self.routes = []
        x = torch.from_numpy(self.embed[tok].astype(np.float32)).to(self.dev)
        dump("embed", x)
        eps = self.c["rms_eps"]
        for i in range(self.L):
            cur = i
            w = self.W[i]
            dump("x_in", x)
            h = rmsnorm(x, w["input_layernorm.weight"].float(), eps)
            x = x + (self.attention(i, h, dump) if self.is_attn[i] else self.deltanet(i, h, dump))
            h = rmsnorm(x, w["post_attention_layernorm.weight"].float(), eps)
            dump("moe_in", h)
            x = x + self.moe(i, h, dump)
        cur = None
        dump("x_final", x)
        logits = self.mv(self.lm_head, rmsnorm(x, self.norm, eps))
        dump("logits", logits)
        self.pos += 1
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--dump", help="directory for golden per-op dumps (every position)")
    ap.add_argument("--routes", help="write per-token routes (jsonl)")
    ap.add_argument("--logits-only", action="store_true", help="with --dump: only the logits (long evaluation sets)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    ids = tk.encode(a.prompt).ids
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
        json.dump({"prompt": a.prompt, "ids": ids}, open(os.path.join(a.dump, "meta.json"), "w"))
    t0 = time.time()
    m = Qwen36Ref()
    m.logits_only = a.logits_only
    print(f"loaded in {time.time() - t0:.1f}s, prompt ids {ids}")
    rf = open(a.routes, "w") if a.routes else None
    out = []
    t0 = time.time()
    for n in range(len(ids) + a.n - 1):
        tok = ids[n] if n < len(ids) else out[-1]
        logits = m.step(tok, a.dump)
        if rf:
            rf.write(json.dumps({"pos": n, "tok": tok, "routes": m.routes}) + "\n")
        if n >= len(ids) - 1:
            out.append(int(logits.argmax()))
            print(repr(tk.decode(out)), flush=True)
    print(f"{(len(ids) + a.n - 1) / (time.time() - t0):.2f} tok/s (reference, not a benchmark)")
    if a.dump:  # every token fed, so an engine can be teacher-forced through the same sequence
        json.dump({"prompt": a.prompt, "ids": ids + out[:-1], "n_prompt": len(ids)}, open(os.path.join(a.dump, "meta.json"), "w"))


if __name__ == "__main__":
    main()
