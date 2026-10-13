"""MTP draft acceptance on the fp32 reference: how often the MTP head's guess for token t+2 equals the main model's
greedy token. Two wirings of the hidden state are measured side by side (pre- vs post-final-norm), since sources
disagree; the right one is the one with high acceptance.

  python tools/mtp_accept.py build/h2h/ids.txt [--prompts 3] [--n 48]
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
from qwen36_ref import Qwen36Ref, rmsnorm

ap = argparse.ArgumentParser()
ap.add_argument("ids")
ap.add_argument("--prompts", type=int, default=3)
ap.add_argument("--n", type=int, default=48)
a = ap.parse_args()

prompts = [[int(x) for x in l.split(",")] for l in open(a.ids) if l.strip()][: a.prompts]
m = Qwen36Ref()
m.logits_only = True
assert m.has_mtp, "no mtp.safetensors in the model folder (tools/fetch_mtp.py)"
L, eps = m.L, m.c["rms_eps"]
hit = {"pre-norm": 0, "post-norm": 0}
total = 0
for pi, ids in enumerate(prompts):
    m.reset()
    draft = {}
    seq = list(ids)
    for pos in range(len(ids) + a.n - 1):
        logits = m.step(seq[pos])
        nxt = int(logits.argmax())
        if pos + 1 >= len(seq): seq.append(nxt)
        if pos >= len(ids) - 1 and pos in draft:  # the draft made at pos-1 guessed this position's next token
            total += 1
            for k, d in draft[pos].items(): hit[k] += d == nxt
        h = m.last_x
        guesses = {}
        for kvi, (name, hh) in enumerate((("pre-norm", h), ("post-norm", rmsnorm(h, m.norm, eps)))):
            guesses[name] = int(m.mtp(seq[pos + 1], hh, pos + 1, L + kvi).argmax())
        draft[pos + 1] = guesses
    print(f"prompt {pi}: " + "  ".join(f"{k} {100 * v / max(total, 1):.1f}%" for k, v in hit.items()) + f"  ({total} drafts)", flush=True)
print("acceptance:", {k: round(v / total, 3) for k, v in hit.items()})
