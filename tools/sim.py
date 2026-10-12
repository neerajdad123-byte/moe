"""Expert-cache policy simulator over real engine route traces (moex --trace).

Replays every (token, layer) routing step with the engine's semantics:
  - CAP slots shared by all layers; a hit runs on the GPU, a miss runs on the CPU
  - after a step, missed experts may be copied into VRAM (admission), at most BW copies per layer step
    (PCIe 4.0 x8 ~13 GB/s over a ~0.5 ms layer step ~= 3-4 copies of 1.77 MB)
  - prefetch for layer l+1 is issued at layer l and shares the same BW budget
  - predictors learn online (counts updated after use), so nothing sees the future except the oracle

  python tools/sim.py build/routes.bin [--cap 1620] [--bw 3]
"""
import argparse, math, sys, time
import numpy as np

L, E, K = 40, 256, 8

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--cap", type=int, default=1620)
ap.add_argument("--bw", type=int, default=3, help="expert copies per layer step (admission + prefetch)")
ap.add_argument("--decode-only", action="store_true", help="score only decode tokens (prompt tokens still warm the cache)")
ap.add_argument("--only", help="run only policies whose name contains this")
a = ap.parse_args()

rec = np.fromfile(a.trace, dtype=np.int16).reshape(-1, 12)
prompt, pos, layer, phase = rec[:, 0].astype(int), rec[:, 1].astype(int), rec[:, 2].astype(int), rec[:, 3].astype(int)
ids = rec[:, 4:].astype(int)
assert (layer == np.tile(np.arange(L), len(rec) // L)).all(), "trace must be whole tokens, layers in order"
T = len(rec) // L
keys = (layer[:, None] * E + ids).reshape(T, L, K)
scored = (phase.reshape(T, L)[:, 0] == 1) if a.decode_only else np.ones(T, bool)
print(f"{T} tokens ({int(scored.sum())} scored) from {prompt.max() + 1} prompts, cap {a.cap} slots, bw {a.bw}/layer step")

# next use (in token index) of each (token, layer, k) access, for the Belady oracle
nxt = np.full((T, L, K), 1 << 30, dtype=np.int64)
last_seen = {}
for t in range(T - 1, -1, -1):
    for l in range(L):
        for k in range(K):
            key = keys[t, l, k]
            nxt[t, l, k] = last_seen.get(key, 1 << 30)
        for k in range(K):
            last_seen[keys[t, l, k]] = t


def run(name, evict, admit_filter=False, tau=None, prefetch=0):
    """evict: 'lru' | 'freq' | 'belady'. tau: decay (tokens) for 'freq'. prefetch: top-M next-layer Markov."""
    if a.only and a.only not in name: return
    slot_of = np.full(L * E, -1, np.int64)
    skey = np.full(a.cap, -1, np.int64)
    sscore = np.full(a.cap, -np.inf)             # higher = keep
    G = np.full(L * E, -np.inf)                   # log of decayed frequency (scaled by exp(t/tau)) of every expert
    trans = np.zeros((L, E, E), np.float32) if prefetch else None  # trans[l, e, e'] = count of e@l -> e'@l+1
    hits = miss = copies = pf_used = pf_issued = 0
    prefetched = set()

    def score(key, t, l):
        if evict == "lru": return t * L + l
        if evict == "freq": return G[key]
        return -oracle_next[key]                  # belady: soonest next use is most valuable

    oracle_next = np.full(L * E, 1 << 30, np.int64)

    def insert(key, t, l, protect):
        nonlocal copies
        if slot_of[key] >= 0: return False
        sc = sscore.copy()
        sc[protect] = np.inf
        v = int(np.argmin(sc))
        if admit_filter and skey[v] >= 0 and score(key, t, l) <= sscore[v]: return False
        if skey[v] >= 0: slot_of[skey[v]] = -1
        skey[v] = key; slot_of[key] = v; sscore[v] = score(key, t, l)
        copies += 1
        return True

    for t in range(T):
        for l in range(L):
            ks = keys[t, l]
            if tau: G[ks] = np.logaddexp(G[ks], t / tau)
            oracle_next[ks] = nxt[t, l]
            sl = slot_of[ks]
            h = sl >= 0
            if scored[t]:
                hits += int(h.sum()); miss += int((~h).sum())
                pf_used += sum(1 for kk in ks[h] if kk in prefetched)
            for kk in ks[h]: prefetched.discard(kk)
            for s_ in sl[h]: sscore[s_] = score(skey[s_], t, l)
            protect = sl[h].tolist()
            budget = a.bw
            for kk in ks[~h]:                     # admission of this step's misses
                if budget == 0: break
                if insert(kk, t, l, protect): budget -= 1; protect.append(slot_of[kk])
            if prefetch and l + 1 < L and budget > 0:
                pred = trans[l, ks - l * E].sum(0)
                for e2 in np.argsort(-pred)[:prefetch]:
                    if budget == 0 or pred[e2] <= 0: break
                    k2 = (l + 1) * E + int(e2)
                    if slot_of[k2] < 0 and insert(k2, t, l, protect):
                        budget -= 1; protect.append(slot_of[k2]); prefetched.add(k2); pf_issued += 1
            if prefetch and l > 0:
                prev = keys[t, l - 1] - (l - 1) * E
                trans[l - 1][np.ix_(prev, ks - l * E)] += 1
    n = int(scored.sum())
    print(f"  {name:38s} hit {100 * hits / (hits + miss):5.1f}%  CPU misses/tok {miss / n:5.1f}  copies/tok {copies / T:5.1f}"
          + (f"  prefetch useful {100 * pf_used / max(pf_issued, 1):4.1f}%" if prefetch else ""), flush=True)


t0 = time.time()
run("LRU (engine today)", "lru")
for tau in (3, 5, 10, 20, 40, 80):
    run(f"decayed-freq tau={tau}", "freq", tau=tau)
for tau in ():
    run(f"decayed-freq tau={tau} + admit filter", "freq", admit_filter=True, tau=tau)
run("Belady oracle (ceiling)", "belady")
for m in (2, 4):
    run(f"LRU + Markov prefetch top-{m}", "lru", prefetch=m)
print(f"({time.time() - t0:.0f} s)")
