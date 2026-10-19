"""Offline GPTQ / AWQ quantizer for the dense trunk -> packed int4 / int6 that moex loads with MOEX_DENSE (research).

  1. calibration inputs (prefill dumps every dense matrix's input rows, fp16; ~0.7 MB per kept row):
       python tools/calib_prompts.py build/calib_ids.txt              (~200 short chat + code prompts, chat template)
       mkdir build/calib; MOEX_CALIB=build/calib MOEX_CALIB_EVERY=4 moex --batch build/calib_ids.txt --n 1 --no-mtp
  2. quantize one group (or several) into a directory:
       python tools/quant_dense.py --calib build/calib --out build/q/gptq4_dn_out --groups dn_out --bits 4 --method gptq --act-order
  3. screen: tools/screen_dense.py (MOEX_DENSE=dir moex --check ...)
  python tools/quant_dense.py --selftest     (CPU only, synthetic data; also writes a file for build/test_dense.exe)

Methods (all on gs-wide groups along the input, gs 32 or 64; symmetric, or asymmetric with a zero point per group):
  rtn       the engine's rounding (symmetric: scale = group absmax / qmax, as quantize_rows_q4_k / _q6_k)
  awq       AWQ (Lin et al. 2023): per input channel scale s = m^alpha (m = rms of the channel over the calibration
            rows), alpha from a grid minimizing the output error tr(E H E^T), then a per-group clip search; the codes
            encode W diag(s), stored with s (N.isc), so w = value / s. A kernel would scale x by 1/s (or fold it into the
            preceding op where one exists).
  gptq      GPTQ (Frantar et al. 2022): columns in order (or by decreasing H_ii with --act-order: static groups,
            scales fixed from W first), each column's rounding error spread over the later columns via the Cholesky
            factor of H^-1. --clip: activation-weighted clip search for each group's scale.
  awq+gptq  AWQ's scaling first, then GPTQ on W diag(s) with H scaled to match.
H = sum of x x^T over the calibration rows; "rel err" = tr(E H E^T) / tr(W H W^T), the relative squared output error.
Output: OUT/dense.safetensors, per container tensor N: N.s f32 [O][I/gs], N.q4 u8 [O][I/2] or N.q6 + N.q6h, N.z f32
[O][I/gs] (asymmetric), N.isc f32 [I] (AWQ): the layout src/dense_ovr.h decodes.
"""
import argparse, os, sys, time
import numpy as np
import torch

# group -> (part suffixes under the layer prefix, layer kind); the row order is wmat's in src/moex.cu
GROUPS = {
    "dn_in": (["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b"], "dn"),
    "dn_out": (["linear_attn.out_proj"], "dn"),
    "attn_qkv": (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "attn"),
    "attn_o": (["self_attn.o_proj"], "attn"),
    "sh_gu": (["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj"], "any"),
    "sh_d": (["mlp.shared_expert.down_proj"], "any"),
    "lm_head": (None, "global"),
}
CLIP_GRID = [1.0 - 0.02 * i for i in range(16)]
ALPHA_GRID = [i / 20 for i in range(21)]


class Q:
    """bits, group size, symmetric or asymmetric."""
    def __init__(self, bits, gs, asym):
        self.bits, self.gs, self.asym = bits, gs, asym
        self.qmax = (1 << bits) - 1 if asym else (1 << (bits - 1)) - 1

    def params(self, w, wt=None, clip=False):
        """w [O, g] -> scale, zero [O, 1] (zero None if symmetric); clip: the ratio minimizing sum wt (w - deq)^2."""
        def make(r):
            if self.asym:
                lo, hi = w.amin(1, keepdim=True) * r, w.amax(1, keepdim=True) * r
                s = torch.where(hi > lo, (hi - lo) / self.qmax, torch.ones_like(hi))
                return s, torch.clamp(torch.round(-lo / s), 0, self.qmax)
            am = w.abs().amax(1, keepdim=True) * r
            return torch.where(am > 0, am / self.qmax, torch.ones_like(am)), None
        if not clip:
            return make(1.0)
        wt = torch.ones(w.shape[1]) if wt is None else wt
        best, berr = None, None
        for r in CLIP_GRID:
            s, z = make(r)
            err = (((w - self.deq(self.code(w, s, z), s, z)) ** 2) * wt).sum(1, keepdim=True)
            if best is None:
                best, berr = (s, z), err
            else:
                t = err < berr
                best = (torch.where(t, s, best[0]), None if z is None else torch.where(t, z, best[1]))
                berr = torch.where(t, err, berr)
        return best

    def code(self, w, s, z):  # round half to even (rintf / __float2int_rn)
        if self.asym:
            return torch.clamp(torch.round(w / s) + z, 0, self.qmax)
        return torch.clamp(torch.round(w / s), -self.qmax - 1, self.qmax)

    @staticmethod
    def deq(c, s, z):
        return (c - z) * s if z is not None else c * s


def expand(M, gs):  # [O, G] -> [O, G * gs]
    return M.repeat_interleave(gs, dim=1)


def dequant(C, S, Z, isc=None):
    gs = C.shape[1] // S.shape[1]
    W = Q.deq(C, expand(S, gs), None if Z is None else expand(Z, gs))
    return W / isc if isc is not None else W


def rtn(W, q, hdiag=None, clip=False):
    O, I = W.shape
    C = torch.empty_like(W); S = torch.empty(O, I // q.gs); Z = torch.empty(O, I // q.gs) if q.asym else None
    for g in range(I // q.gs):
        sl = slice(g * q.gs, (g + 1) * q.gs)
        s, z = q.params(W[:, sl], None if hdiag is None else hdiag[sl], clip)
        C[:, sl] = q.code(W[:, sl], s, z); S[:, g] = s[:, 0]
        if q.asym: Z[:, g] = z[:, 0]
    return C, S, Z


def gptq(W, H, q, clip=False, act_order=False, damp=0.01, block=128):
    """W [O, I] fp32, H [I, I] -> codes, scales, zeros. block is a multiple of gs: in natural order a group never
    straddles two blocks, so its scale is picked from the error-updated weights when GPTQ reaches it."""
    W = W.clone(); H = H.clone().double()
    O, I = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(I, dtype=H.dtype)
    hd = torch.diag(H).float()
    G = I // q.gs
    S = torch.empty(O, G); Z = torch.empty(O, G) if q.asym else None
    if act_order:  # static groups: every group's scale from W before any update, then columns by decreasing H_ii
        _, S, Z = rtn(W, q, hd, clip)
        perm = torch.argsort(hd, descending=True)
    else:
        perm = torch.arange(I)
    W = W[:, perm]; H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    C = torch.empty_like(W)
    for i1 in range(0, I, block):
        i2 = min(i1 + block, I)
        W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            g = int(perm[c]) // q.gs
            if not act_order and c % q.gs == 0:
                s, z = q.params(W1[:, i:i + q.gs], hd[c:c + q.gs], clip)
                S[:, g] = s[:, 0]
                if q.asym: Z[:, g] = z[:, 0]
            s = S[:, g:g + 1]; z = Z[:, g:g + 1] if q.asym else None
            code = q.code(W1[:, i:i + 1], s, z)
            C[:, c] = code[:, 0]
            e = (W1[:, i] - q.deq(code, s, z)[:, 0]) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    Cn = torch.empty_like(C); Cn[:, perm] = C  # back to the natural column order
    return Cn, S, Z


def rel_err(W, Wq, H):
    E = (Wq - W).double(); Wd = W.double(); Hd = H.double()
    return float(((E @ Hd) * E).sum() / ((Wd @ Hd) * Wd).sum())


def awq_scales(W, H, n, q, rows=16384):
    """AWQ's scale search on (up to `rows` evenly spaced rows of) W: s = m^alpha, m = per-channel rms."""
    # shortcut: alpha is chosen on a row subset of big matrices (lm_head: 248K rows); all rows if quality demands
    Ws = W[np.linspace(0, W.shape[0] - 1, min(rows, W.shape[0])).astype(np.int64)] if W.shape[0] > rows else W
    m = torch.sqrt(torch.clamp(torch.diag(H) / max(n, 1), min=1e-12))
    best, berr = torch.ones(W.shape[1]), None
    for a in ALPHA_GRID:
        s = m ** a
        s = s / torch.sqrt(s.max() * s.min())  # keep the scaled weights near the original magnitude
        err = rel_err(Ws, dequant(*rtn(Ws * s, q), isc=s), H)
        if berr is None or err < berr: best, berr = s, err
    return best


def quantize(W, H, n, q, method, clip=False, act_order=False, damp=0.01, rows_chunk=16384):
    """-> codes, scales, zeros (or None), isc (or None)"""
    isc = None
    if method in ("awq", "awq+gptq"):
        isc = awq_scales(W, H, n, q)
        W = W * isc
        H = H / isc[:, None] / isc[None, :]  # x' = x / s: the same outputs
    O, I = W.shape
    C = torch.empty_like(W); S = torch.empty(O, I // q.gs); Z = torch.empty(O, I // q.gs) if q.asym else None
    for r0 in range(0, O, rows_chunk):  # rows are independent given H
        r1 = min(r0 + rows_chunk, O)
        if method in ("gptq", "awq+gptq"): c, s, z = gptq(W[r0:r1], H, q, clip, act_order, damp)
        else: c, s, z = rtn(W[r0:r1], q, torch.diag(H), clip or method == "awq")
        C[r0:r1], S[r0:r1] = c, s
        if q.asym: Z[r0:r1] = z
    return C, S, Z, isc


def pack(C, q):
    """codes -> planes: symmetric q4 two's complement nibbles, q6 u = c + 32; asymmetric: the unsigned codes."""
    c = C.to(torch.int32).numpy()
    u = (c if q.asym else (c & 15 if q.bits == 4 else c + 32)).astype(np.uint8)
    if q.bits == 4:
        return {"q4": (u[:, 0::2] & 15) | ((u[:, 1::2] & 15) << 4)}
    lo, hi = u & 15, u >> 4
    return {"q6": lo[:, 0::2] | (lo[:, 1::2] << 4),
            "q6h": hi[:, 0::4] | (hi[:, 1::4] << 2) | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6)}


def tensors(name, C, S, Z, isc, q):
    out = {f"{name}.{k}": np.ascontiguousarray(v) for k, v in pack(C, q).items()}
    out[f"{name}.s"] = S.numpy().astype(np.float32)
    if Z is not None: out[f"{name}.z"] = Z.numpy().astype(np.float32)
    if isc is not None: out[f"{name}.isc"] = isc.numpy().astype(np.float32)
    return out


def load_h(path, I, max_rows):
    X = np.fromfile(path, dtype=np.float16).reshape(-1, I)
    if max_rows and len(X) > max_rows:
        X = X[np.linspace(0, len(X) - 1, max_rows).astype(np.int64)]
    H = torch.zeros(I, I, dtype=torch.float64)
    for r in range(0, len(X), 8192):
        x = torch.from_numpy(X[r:r + 8192].astype(np.float32)).double()
        H += x.T @ x
    return H.float(), len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib"); ap.add_argument("--out")
    ap.add_argument("--groups", default="dn_out"); ap.add_argument("--bits", type=int, default=4, choices=[4, 6])
    ap.add_argument("--gs", type=int, default=64, choices=[32, 64]); ap.add_argument("--asym", action="store_true")
    ap.add_argument("--method", default="gptq", choices=["rtn", "awq", "gptq", "awq+gptq"])
    ap.add_argument("--act-order", action="store_true"); ap.add_argument("--clip", action="store_true")
    ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--layers", default="", help="e.g. 0-9 (default all 40)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--model", default=os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.threads: torch.set_num_threads(a.threads)
    if a.selftest: return selftest()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref"))
    from qwen36_ref import Container
    from safetensors.numpy import save_file
    C = Container(a.model)
    q = Q(a.bits, a.gs, a.asym)
    os.makedirs(a.out, exist_ok=True)
    layers = range(40)
    if a.layers:
        lo, _, hi = a.layers.partition("-"); layers = range(int(lo), int(hi or lo) + 1)
    out, t0, log = {}, time.time(), []
    for g in a.groups.split(","):
        parts, kind = GROUPS[g]
        jobs = [(-1, ["lm_head.weight"], "lm_head.f16")] if kind == "global" else [
            (l, [f"model.layers.{l}.{p}.weight" for p in parts], f"{g}.L{l}.f16") for l in layers
            if f"model.layers.{l}.{parts[0]}.weight" in C.index]
        for l, names, cf in jobs:
            W = torch.cat([torch.from_numpy(np.array(C.np(nm), dtype=np.float32)) for nm in names])
            H, n = load_h(os.path.join(a.calib, cf), W.shape[1], a.max_rows)
            Cq, S, Z, isc = quantize(W, H, n, q, a.method, a.clip, a.act_order, a.damp)
            e0, e1 = rel_err(W, dequant(*rtn(W, Q(a.bits, 64, False))[:2], None), H), rel_err(W, dequant(Cq, S, Z, isc), H)
            line = f"{g} L{l}: {n} rows, rel err rtn gs64 {e0:.3e} -> {a.method} {e1:.3e}  ({time.time() - t0:.0f} s)"
            print(line, flush=True); log.append(line)
            off = 0
            for nm in names:
                rows = C.index[nm][2]["shape"][0]
                out.update(tensors(nm, Cq[off:off + rows], S[off:off + rows], None if Z is None else Z[off:off + rows], isc, q))
                off += rows
    meta = {k: str(v) for k, v in vars(a).items() if k not in ("selftest", "threads")}
    save_file(out, os.path.join(a.out, "dense.safetensors"), metadata=meta)
    open(os.path.join(a.out, "log.txt"), "w").write("\n".join(log) + "\n")
    print("wrote", os.path.join(a.out, "dense.safetensors"))


def selftest():
    torch.manual_seed(0)
    O, I, N = 96, 512, 2048
    # correlated activations with a few outlier channels (what makes GPTQ / AWQ matter)
    X = torch.randn(N, I) @ (torch.eye(I) + 0.3 * torch.randn(I, I) / I ** 0.5)
    X[:, :8] *= 12
    W = torch.randn(O, I) * 0.02
    W[:, 100:104] += 0.08  # an offset: asymmetric groups have something to gain
    H = X.T @ X
    for bits in (4, 6):
        q = Q(bits, 64, False)
        Cr, Sr, _ = rtn(W, q)
        am = W.view(O, -1, 64).abs().amax(2)  # the engine's rounding
        assert torch.allclose(Sr, am / q.qmax) and Cr.min() >= -q.qmax - 1 and Cr.max() <= q.qmax
        e = {}
        for name, qq, method, kw in [("rtn", q, "rtn", {}), ("awq", q, "awq", {}), ("gptq", q, "gptq", {}),
                                     ("gptq act", q, "gptq", {"act_order": True}), ("gptq clip", q, "gptq", {"clip": True}),
                                     ("awq+gptq", q, "awq+gptq", {}), ("gptq gs32", Q(bits, 32, False), "gptq", {}),
                                     ("gptq asym", Q(bits, 64, True), "gptq", {})]:
            Cq, S, Z, isc = quantize(W, H, N, qq, method, **kw)
            e[name] = rel_err(W, dequant(Cq, S, Z, isc), H)
        print(f"int{bits} rel output err: " + "  ".join(f"{k} {v:.2e}" for k, v in e.items()))
        assert e["awq"] <= e["rtn"] and e["awq+gptq"] < e["rtn"]
        assert all(e[k] < 0.8 * e["rtn"] for k in ("gptq", "gptq act", "gptq clip", "gptq gs32"))
        assert e["gptq gs32"] < e["gptq"]
    q = Q(4, 64, False)  # identity Hessian: GPTQ has nothing to trade, so it equals rtn
    Cg, Sg, _ = gptq(W, torch.eye(I) * N, q, damp=0.0)
    Cr, Sr, _ = rtn(W, q)
    assert torch.equal(Cg, Cr) and torch.allclose(Sg, Sr)
    # a file in every layout for tests/test_dense.cpp: the C++ decoder must give exactly these values
    from safetensors.numpy import save_file
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "dense_test")
    os.makedirs(d, exist_ok=True)
    out, ref = {}, []
    for i, (bits, gs, asym, method) in enumerate([(4, 64, False, "rtn"), (6, 64, False, "gptq"), (4, 32, True, "awq"), (6, 32, True, "awq+gptq")]):
        qq = Q(bits, gs, asym)
        Cq, S, Z, isc = quantize(W, H, N, qq, method)
        out.update(tensors(f"t{i}", Cq, S, Z, isc, qq))
        ref.append(dequant(Cq, S, Z, isc).numpy().astype(np.float32))
    save_file(out, os.path.join(d, "dense.safetensors"))
    np.concatenate(ref).tofile(os.path.join(d, "ref.f32"))
    print(f"selftest PASS (wrote {d} for test_dense: 4 tensors {O}x{I})")


if __name__ == "__main__":
    main()
