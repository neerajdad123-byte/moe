#!/usr/bin/env python3
"""MoEx numpy reference forward pass for Qwen3-30B-A3B (G1 correctness anchor).

Independent clean-room implementation of the Qwen3-MoE decode math in numpy,
using the pure-numpy gguf dequant as the weight source. Produces a golden trace
(per-layer checkpoints, router IDs/weights, final logits, argmax token) for a
FIXED input token sequence, so the C++/CUDA path can be validated against it
without a tokenizer.

Architecture (from Qwen/Qwen3-30B-A3B config):
  48 layers, d_model=2048, 32 Q heads / 4 KV heads, head_dim=128,
  per-head q/k RMSNorm, RoPE theta=1e6, 128 experts top-8 softmax+norm,
  SwiGLU expert FFN d_ff=768, RMSNorm eps=1e-6, no shared expert, MoE every layer.

Output: build/ref/trace.npz  and  build/ref/summary.txt
"""
import os, sys, time
import numpy as np
from gguf import GGUFReader
import gguf.quants as q

BLOB = sys.argv[1] if len(sys.argv) > 1 else r"C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
INPUT_IDS = [151643, 9707, 11, 1879, 0, 358, 1079]  # fixed arbitrary prompt ids
OUT = os.path.join("build", "ref"); os.makedirs(OUT, exist_ok=True)

D_MODEL=2048; N_LAYERS=48; N_HEAD=32; N_KV=4; HEAD_DIM=128
N_EXP=128; TOPK=8; D_FF=768; EPS=1e-6; ROPE_BASE=1e6

print("loading gguf..."); t0=time.time()
r = GGUFReader(BLOB)
T = {t.name: t for t in r.tensors}
print(f"  {len(T)} tensors in {time.time()-t0:.1f}s")

_cache={}
def W(name):
    """Dequantized f32 tensor with logical shape (rows, cols) = (dim1, dim0)."""
    if name in _cache: return _cache[name]
    t=T[name]
    arr=q.dequantize(t.data, t.tensor_type).astype(np.float32)
    # gguf data shape is reversed dims; reshape to (slow..fast)
    arr=arr.reshape(tuple(int(x) for x in reversed(t.shape)))
    _cache[name]=arr
    return arr

def rmsnorm(x, w, eps=EPS):
    v=np.mean(x*x, axis=-1, keepdims=True)
    return x/np.sqrt(v+eps)*w

def silu(x): return x/(1.0+np.exp(-x))

def softmax(x):
    m=np.max(x,axis=-1,keepdims=True); e=np.exp(x-m); return e/np.sum(e,axis=-1,keepdims=True)

# RoPE tables
def rope_cos_sin(pos, dim=HEAD_DIM, base=ROPE_BASE):
    inv=1.0/(base**(np.arange(0,dim,2)/dim))
    ang=pos*inv                      # (dim/2,)
    return np.cos(ang), np.sin(ang)

def apply_rope(vec, cos, sin):
    # vec: (..., head_dim). Qwen/GGUF uses split-half (NEOX) rotation.
    half=vec.shape[-1]//2
    a=vec[...,:half]; b=vec[...,half:]
    return np.concatenate([a*cos - b*sin, b*cos + a*sin], axis=-1)

ids=np.array(INPUT_IDS, dtype=np.int64); S=len(ids)
print(f"forward: {S} tokens")

emb=W("token_embd.weight")            # (vocab, d_model)
x=emb[ids].astype(np.float32).copy()  # (S, d_model)

ckpt={"input_ids":ids, "embed":x.copy()}
router_ids=np.zeros((N_LAYERS,S,TOPK),dtype=np.int64)
router_wts=np.zeros((N_LAYERS,S,TOPK),dtype=np.float32)

for L in range(N_LAYERS):
    p=f"blk.{L}."
    h=rmsnorm(x, W(p+"attn_norm.weight"))
    Wq=W(p+"attn_q.weight"); Wk=W(p+"attn_k.weight"); Wv=W(p+"attn_v.weight")
    qh=(h@Wq.T).reshape(S,N_HEAD,HEAD_DIM)
    kh=(h@Wk.T).reshape(S,N_KV,HEAD_DIM)
    vh=(h@Wv.T).reshape(S,N_KV,HEAD_DIM)
    qn=W(p+"attn_q_norm.weight"); kn=W(p+"attn_k_norm.weight")
    qh=rmsnorm(qh, qn); kh=rmsnorm(kh, kn)
    # RoPE per position
    for s in range(S):
        c,sn=rope_cos_sin(s)
        qh[s]=apply_rope(qh[s],c,sn); kh[s]=apply_rope(kh[s],c,sn)
    # GQA causal attention
    grp=N_HEAD//N_KV; scale=1.0/np.sqrt(HEAD_DIM)
    out=np.zeros((S,N_HEAD,HEAD_DIM),dtype=np.float32)
    for hd in range(N_HEAD):
        kv=hd//grp
        Q=qh[:,hd,:]; K=kh[:,kv,:]; V=vh[:,kv,:]
        att=(Q@K.T)*scale
        mask=np.triu(np.ones((S,S)),1).astype(bool); att[mask]=-np.inf
        out[:,hd,:]=softmax(att)@V
    ao=out.reshape(S,N_HEAD*HEAD_DIM)@W(p+"attn_output.weight").T
    x=x+ao
    if L==0: ckpt["l0_attn_out"]=x.copy()
    # MoE FFN
    h=rmsnorm(x, W(p+"ffn_norm.weight"))
    logits=h@W(p+"ffn_gate_inp.weight").T          # (S, n_exp)
    probs=softmax(logits)
    ge=W(p+"ffn_gate_exps.weight"); ue=W(p+"ffn_up_exps.weight"); de=W(p+"ffn_down_exps.weight")
    moe=np.zeros_like(x)
    for s in range(S):
        top=np.argsort(-probs[s])[:TOPK]
        w=probs[s][top]; w=w/np.sum(w)
        router_ids[L,s]=top; router_wts[L,s]=w
        for k in range(TOPK):
            e=top[k]
            g=silu(h[s]@ge[e].T); u=h[s]@ue[e].T
            moe[s]+=w[k]*(de[e]@(g*u))
    x=x+moe
    if L==0: ckpt["l0_ffn_out"]=x.copy()
    if L%8==0: print(f"  layer {L} done ({time.time()-t0:.0f}s)")

x=rmsnorm(x, W("output_norm.weight"))
logits=x@W("output.weight").T          # (S, vocab)
argmax=np.argmax(logits,axis=-1)
ckpt["final_logits"]=logits.astype(np.float32)
ckpt["argmax"]=argmax

np.savez(os.path.join(OUT,"trace.npz"),
         router_ids=router_ids, router_wts=router_wts, **ckpt)
with open(os.path.join(OUT,"summary.txt"),"w") as f:
    f.write(f"input_ids={list(ids)}\n")
    f.write(f"argmax={list(argmax)}\n")
    f.write(f"last_tok_top5={list(np.argsort(-logits[-1])[:5])}\n")
    f.write(f"l0 router ids (tok0)={list(router_ids[0,0])}\n")
    f.write(f"l0 router wts (tok0)={list(router_wts[0,0])}\n")
    f.write(f"final_logit[-1][:5]={list(logits[-1][:5])}\n")
print("SUMMARY:"); print(open(os.path.join(OUT,"summary.txt")).read())
print(f"done in {time.time()-t0:.0f}s -> {OUT}/trace.npz")
