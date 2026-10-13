"""Would MTP (multi-token prediction, 1 draft token per step) speed up moex? Answered on real route traces, no model.

With MTP each decode step runs 2 tokens through the model (the next token + a draft). Dense weights are read once for
both, but each layer needs the UNION of both tokens' experts, so more of them miss the VRAM cache and run on the CPU.
This replays the trace with the engine's cache semantics (sim.py: decayed-frequency eviction tau=10, cap slots, bw
admissions per layer step) and a per-layer time model calibrated to the measured engine:

  layer time = D * (1 + d2*(n-1)) + max(GPU hits, CPU misses)      (hits and misses overlap, as in the engine)
  GPU hits   = g * sum over hit experts of (1 + bg*(users-1))
  CPU misses = c * sum over missed experts of (1 + bc*(users-1))
  step       = layers * (1 + 1/L for the MTP head, ~one more layer) + state snapshot for rollback

D is solved so that 1 token/step reproduces the measured 46.2 tok/s. c = 6.5 ms / 92 misses (measured), g = 1.77 MB at
~150 GB/s. The draft token's routing is approximated by the true next token's (exact when the draft is accepted).

  python tools/mtp_sim.py build/routes_v2.bin
"""
import argparse
import numpy as np

L, E, K = 40, 256, 8
ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--cap", type=int, default=1588)
ap.add_argument("--bw", type=int, default=3)
ap.add_argument("--tps", type=float, default=46.2, help="measured decode tok/s at 1 token/step")
a = ap.parse_args()

rec = np.fromfile(a.trace, dtype=np.int16).reshape(-1, 12)
T = len(rec) // L
keys = (rec[:, 2].astype(np.int64)[:, None] * E + rec[:, 4:].astype(np.int64)).reshape(T, L, K)
prompt = rec[::L, 0].astype(int)
decode = rec[::L, 3] == 1

c_ms, g_ms = 6.5 / 92, 1.77e6 / 150e9 * 1e3


def replay(alpha, n_draft, bc, bg, seed=0):
    """Cache replay. Returns per decode step (sum over layers of max(gpu, cpu) ms, tokens in step, tokens accepted),
    plus hit rate and misses per step. n_draft=0: today's engine."""
    rng = np.random.default_rng(seed)
    slot_of = np.full(L * E, -1, np.int64)
    skey = np.full(a.cap, -1, np.int64)
    sscore = np.full(a.cap, -np.inf)
    G = np.full(L * E, -np.inf)
    tau = 10.0
    t, hits, miss, out = 0, 0, 0, []
    while t < T:
        span = [t] if not decode[t] else [u for u in range(t, min(t + 1 + n_draft, T)) if decode[u] and prompt[u] == prompt[t]]
        M = 0.0
        for l in range(L):
            for u in span: G[keys[u, l]] = np.logaddexp(G[keys[u, l]], u / tau)
            uk, users = np.unique(keys[span, l].ravel(), return_counts=True)
            sl = slot_of[uk]
            h = sl >= 0
            for s_ in sl[h]: sscore[s_] = G[skey[s_]]
            protect = sl[h].tolist()
            budget = a.bw
            for kk in uk[~h]:
                if budget == 0: break
                sc = sscore.copy(); sc[protect] = np.inf
                v = int(np.argmin(sc))
                if skey[v] >= 0: slot_of[skey[v]] = -1
                skey[v] = kk; slot_of[kk] = v; sscore[v] = G[kk]; protect.append(v); budget -= 1
            if decode[t]:
                M += max(g_ms * (1 + bg * (users[h] - 1)).sum(), c_ms * (1 + bc * (users[~h] - 1)).sum())
                hits += int(h.sum()); miss += int((~h).sum())
        if not decode[t]:
            t += 1; continue
        adv = 1
        for _ in span[1:]:  # accept each draft in order with probability alpha
            if rng.random() < alpha: adv += 1
            else: break
        out.append((M, len(span), adv)); t += adv
    return np.array(out), hits / max(hits + miss, 1), miss / max(len(out), 1)


def tok_s(steps, D, d2=0.1, snapshot_ms=0.4):
    M, n, adv = steps[:, 0], steps[:, 1], steps[:, 2]
    mtp = n > 1
    ms = (L * D * (1 + d2 * (n - 1)) + M) * (1 + mtp / L) + snapshot_ms * mtp
    return adv.sum() / ms.sum() * 1e3, ms.mean()


base, bh, bm = replay(0, 0, 0, 0)
D = (len(base) / a.tps * 1e3 - base[:, 0].sum()) / (len(base) * L)  # today: total ms = steps*L*D + sum(M)
print(f"{T} tokens ({len(base)} decode steps), cap {a.cap}, bw {a.bw}. Calibrated dense+overhead per layer "
      f"D = {D:.3f} ms; c = {c_ms:.3f} ms per CPU expert, g = {g_ms:.3f} ms per GPU expert")
bt = tok_s(base, D)
print(f"today (1 token/step): {bt[0]:.1f} tok/s, hit {100 * bh:.1f}%, {bm:.0f} CPU misses/step, {bt[1]:.1f} ms/step")
print("\nMTP, 1 draft/step. bc = extra CPU cost of a 2nd token on the same expert (0.3 = batched kernel, 1.0 = none)")
print(f"{'accept':>6} {'bc':>4} | {'tok/s':>6} {'vs today':>8} | {'hit':>6} {'misses/step':>11} {'ms/step':>7}")
for bc in (0.3, 1.0):
    for alpha in (0.6, 0.7, 0.8, 0.9):
        st, h, m = replay(alpha, 1, bc, 0.2)
        r = tok_s(st, D)
        print(f"{alpha:6.1f} {bc:4.1f} | {r[0]:6.1f} {r[0] / bt[0]:7.2f}x | {100 * h:5.1f}% {m:11.0f} {r[1]:7.1f}", flush=True)
