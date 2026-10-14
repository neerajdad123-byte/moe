"""16 GB machines: RAM tier policies + layer-ahead SSD prefetch, on real route traces, with measured SSD timings.

Tiers: VRAM (engine policy: decayed-frequency tau=10, cap slots, bw admissions per layer step), RAM (R experts),
SSD. A VRAM miss runs on the CPU and needs the expert in RAM; if it isn't there the token waits for an SSD read
(demand read, critical path). Background prefetch reads happen while layers compute, within the SSD's throughput.

Measured (tools/ssd_bench.py, this laptop): one cold read 1.68 ms; with 8+ in flight 0.46 ms/expert (3.8 GB/s).
Token time without SSD waits: 21.6 ms (46.2 tok/s), i.e. 0.54 ms per layer -> ~1.17 background reads per layer.

RAM policies:
  lru          recency, what Windows' file cache does today (demand reads only)
  static X     the X most-used experts (counted on prompts 0-39) locked in RAM, the rest always from SSD
  lru+pf k     lru + at layer l, prefetch experts predicted for layer l+k (lag-k co-occurrence with the experts the
               token picked at layer l, learned online) into RAM, within the background budget
Scored on the decode tokens of prompts 40-79.

  python tools/prefetch_sim.py build/routes_v2.bin
"""
import argparse
from collections import OrderedDict
import numpy as np

L, E, K = 40, 256, 8
EXPERT_GB = 1.769472e-3
ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--cap", type=int, default=1623)
ap.add_argument("--bw", type=int, default=3)
ap.add_argument("--t-token", type=float, default=21.6)
ap.add_argument("--t-read", type=float, default=1.68, help="ms, one demand read")
ap.add_argument("--t-par", type=float, default=0.46, help="ms per expert with many reads in flight")
a = ap.parse_args()

rec = np.fromfile(a.trace, dtype=np.int16).reshape(-1, 12)
T = len(rec) // L
keys = (rec[:, 2].astype(np.int64)[:, None] * E + rec[:, 4:].astype(np.int64)).reshape(T, L, K)
prompt, decode = rec[::L, 0].astype(int), rec[::L, 3] == 1
train = prompt <= prompt.max() // 2
counts = np.bincount(keys[train].ravel(), minlength=L * E)
t_layer = a.t_token / L
budget_per_layer = t_layer / a.t_par


def demand_ms(k):  # k demand reads in one layer, issued together
    return 0.0 if k == 0 else a.t_read + (k - 1) * a.t_par


def run(R, policy, lag=0, topm=4):
    slot_of = np.full(L * E, -1, np.int64); skey = np.full(a.cap, -1, np.int64); sscore = np.full(a.cap, -np.inf)
    G = np.full(L * E, -np.inf)
    ram = OrderedDict()
    static = set(np.argsort(-counts)[:R].tolist()) if policy == "static" else None
    co = np.zeros((L, E, E), np.float32) if lag else None  # co[l, e, e'] : e at layer l -> e' at layer l+lag
    credit = 0.0
    n_tok = stall_ms = demand = pf_reads = pf_used = 0
    pf_set = set()

    def ram_has(k):
        if policy == "static": return k in static
        if k in ram: ram.move_to_end(k); return True
        return False

    def ram_put(k):
        if policy == "static": return
        ram[k] = None; ram.move_to_end(k)
        if len(ram) > R: ram.popitem(last=False)

    for t in range(T):
        score = not train[t] and decode[t]
        tok_stall = 0.0
        for l in range(L):
            ks = keys[t, l]
            G[ks] = np.logaddexp(G[ks], t / 10.0)
            sl = slot_of[ks]; h = sl >= 0
            for s_ in sl[h]: sscore[s_] = G[skey[s_]]
            nd = 0
            for k in ks[~h]:
                k = int(k)
                if ram_has(k):
                    if k in pf_set: pf_used += score; pf_set.discard(k)
                else:
                    nd += 1; ram_put(k)
            tok_stall += demand_ms(nd)
            if score: demand += nd
            protect = sl[h].tolist(); budget = a.bw
            for k in ks[~h]:
                if budget == 0: break
                sc = sscore.copy(); sc[protect] = np.inf; v = int(np.argmin(sc))
                if skey[v] >= 0: slot_of[skey[v]] = -1
                skey[v] = k; slot_of[k] = v; sscore[v] = G[k]; protect.append(v); budget -= 1
            if lag:
                credit = min(credit + budget_per_layer - nd, 4 * budget_per_layer)  # demand reads use the SSD too
                lt = l + lag
                if lt < L:
                    pred = co[l, ks - l * E].sum(0)
                    for e2 in np.argsort(-pred)[:topm]:
                        if credit < 1 or pred[e2] <= 0: break
                        k2 = lt * E + int(e2)
                        if slot_of[k2] >= 0 or k2 in ram: continue
                        ram_put(k2); pf_set.add(k2); credit -= 1; pf_reads += score
                if l >= lag:  # learn: experts at l-lag -> experts at l, this token
                    prev = keys[t, l - lag] - (l - lag) * E
                    co[l - lag][np.ix_(prev, ks - l * E)] += 1
        if score:
            n_tok += 1; stall_ms += tok_stall
    ms = a.t_token + stall_ms / n_tok
    return 1e3 / ms, demand / n_tok, pf_reads / max(n_tok, 1), pf_used / max(n_tok, 1)


order = np.argsort(-counts)
cov = np.cumsum(counts[order]) / counts.sum()
print("Coverage of expert uses by the N most-used experts (prompts 0-39):")
for q in (0.5, 0.8, 0.9, 0.95, 0.99):
    n = int(np.searchsorted(cov, q)) + 1
    print(f"  {int(q * 100):3d}% of uses: {n:5d} experts = {n * EXPERT_GB:5.1f} GB")
print(f"\nSSD: demand read {a.t_read} ms, background {a.t_par} ms/expert -> {budget_per_layer:.2f} prefetches per layer"
      f" ({budget_per_layer * L:.0f} per token); token without SSD waits {a.t_token} ms = {1e3 / a.t_token:.1f} tok/s\n")
print(f"{'RAM for experts':>15} | {'policy':14} | {'tok/s':>6} | {'SSD waits/token':>15} | {'prefetched/token':>16} | {'prefetch hits/token':>19}")
for gb in (6, 8, 10):
    R = int(gb / EXPERT_GB)
    for name, pol, lag in (("lru (today)", "lru", 0), ("static top-R", "static", 0), ("lru+pf lag2", "lru", 2), ("lru+pf lag4", "lru", 4)):
        r = run(R, pol, lag)
        print(f"{gb:8d} GB ({R:5d}) | {name:14} | {r[0]:6.1f} | {r[1]:15.1f} | {r[2]:16.1f} | {r[3]:19.1f}", flush=True)
print(f"\nall experts in RAM (24 GB machine): {1e3 / a.t_token:.1f} tok/s")
