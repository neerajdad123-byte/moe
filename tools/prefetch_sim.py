"""16 GB machines: RAM tier policies + layer-ahead SSD prefetch, on real route traces, with measured SSD timings.

Tiers: VRAM (engine policy: decayed-frequency tau=10, cap slots, bw admissions per layer step), RAM (R experts),
SSD. A VRAM miss runs on the CPU and needs the expert in RAM; if it isn't there the token waits for an SSD read.
The VRAM tier doesn't depend on the RAM tier, so it is replayed once and every RAM policy replays its misses.

Measured (tools/ssd_bench.py, tests/ssd_test.cpp, this laptop): one cold read through the file cache 1.68 ms; a whole
expert as one direct read 0.7-0.9 ms; with 8+ in flight 0.46 ms/expert (3.8 GB/s). Token without SSD waits: 21.6 ms
(46.2 tok/s), i.e. 0.54 ms per layer.

SSD model (all policies but "today"): one FIFO device, a read finishes at max(issue + t_read, device free + t_par).
Prefetches are capped at --qd in flight; a demand read queues behind them (NVMe has no priorities). A token that needs
an expert still in flight waits for it (a late prefetch still saves part of the read).

Policies (RAM for experts = R):
  today          Windows file cache (recency), demand reads only, t_cold + 0.46 ms per extra read in the layer
  own+direct     moex's own RAM cache (recency) + direct reads, demand only
  +ahead k Nc    + at layer l, prefetch the top-Nc experts of layer l+k's router on layer l's hidden state
                 (--ahead file from `moex --batch ... --trace r.bin --trace-ahead a.bin`), skipping ones in VRAM/RAM
  +cycle W       eviction: among the W least recently used, the one whose layer comes up last (just-processed
                 layers first); never an expert in flight or due in the next 4 layers. Larger W: cycle order over recency
  +far D N       no predictor: at layer l, prefetch the N most-used experts (decayed counts, half-life ~70 tokens) of
                 layer (l+D) mod L that are in neither VRAM nor RAM, e.g. at layer 10 load layer 25's
  +victim        VRAM and RAM exclusive: an expert admitted to VRAM leaves RAM, one evicted from VRAM goes back to
                 RAM (copied over PCIe, no SSD read)
Scored on the decode tokens of prompts 40-79.

  python tools/prefetch_sim.py build/routes_v3.bin --ahead build/ahead_v3.bin
"""
import argparse, heapq
from collections import OrderedDict
import numpy as np

L, E, K = 40, 256, 8
EXPERT_GB = 1.769472e-3
ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--ahead", help="router-ahead predictions (moex --trace-ahead)")
ap.add_argument("--cap", type=int, default=1623)
ap.add_argument("--bw", type=int, default=3)
ap.add_argument("--t-token", type=float, default=21.6)
ap.add_argument("--t-cold", type=float, default=1.68, help="ms, one demand read through the file cache (today)")
ap.add_argument("--t-read", type=float, default=0.8, help="ms, one direct read of a whole expert")
ap.add_argument("--t-par", type=float, default=0.46, help="ms per expert with many reads in flight")
ap.add_argument("--qd", type=int, default=8, help="max prefetches in flight")
a = ap.parse_args()

rec = np.fromfile(a.trace, dtype=np.int16).reshape(-1, 12)
T = len(rec) // L
keys = (rec[:, 2].astype(np.int64)[:, None] * E + rec[:, 4:].astype(np.int64)).reshape(T, L, K)
prompt, decode = rec[::L, 0].astype(int), rec[::L, 3] == 1
train = prompt <= prompt.max() // 2
scored = ~train & decode
counts = np.bincount(keys[train].ravel(), minlength=L * E)
t_layer = a.t_token / L
A = np.fromfile(a.ahead, dtype=np.int16).reshape(T, L, 5, 16) if a.ahead else None

# ---- VRAM tier, replayed once: per (t, l) the CPU misses, the experts admitted and the ones they evicted ----
slot_of = np.full(L * E, -1, np.int64); skey = np.full(a.cap, -1, np.int64); sscore = np.full(a.cap, -np.inf)
G = np.full(L * E, -np.inf)
miss, admitted, evicted = {}, {}, {}
for t in range(T):
    for l in range(L):
        ks = keys[t, l]
        G[ks] = np.logaddexp(G[ks], t / 10.0)
        sl = slot_of[ks]; h = sl >= 0
        for s_ in sl[h]: sscore[s_] = G[skey[s_]]
        m = [int(k) for k in ks[~h]]
        protect = sl[h].tolist(); budget = a.bw; ad, ev = [], []
        for k in m:
            if budget == 0: break
            sc = sscore.copy(); sc[protect] = np.inf; v = int(np.argmin(sc))
            if skey[v] >= 0: slot_of[skey[v]] = -1; ev.append(int(skey[v]))
            skey[v] = k; slot_of[k] = v; sscore[v] = G[k]; protect.append(v); budget -= 1; ad.append(k)
        miss[t, l], admitted[t, l], evicted[t, l] = m, ad, ev


def run(R, policy="direct", k_ahead=(), nc=8, cycle_w=0, victim=False, far_d=0, far_n=0, tau=100.0):
    ram = OrderedDict()  # key -> time its data is in RAM (in flight while > now)
    vram = set()
    now = ssd_free = 0.0
    n_tok = stall_tot = demand = pf_reads = pf_hits = pf_late = 0
    pf_set = set()
    flight = []  # finish times of prefetches, for the --qd cap
    F = np.full(L * E, -np.inf)  # decayed use counts (log), for the far-ahead prefetch

    def ssd_read():
        nonlocal ssd_free
        start = max(ssd_free, now)
        done = max(now + a.t_read, start + a.t_par)
        ssd_free = start + a.t_par
        return done

    def evict_one(l):
        if cycle_w:
            best, bd = None, -1
            for i, (k, ready) in enumerate(ram.items()):
                if i >= cycle_w: break
                if ready > now: continue
                d = (k // E - l - 1) % L  # L-1: this layer (next use furthest away), 0: the next layer
                if d < 4: continue        # coming up in the next layers: predicted or about to be needed
                if d > bd: best, bd = k, d
            if best is not None: del ram[best]; return
        for k, ready in ram.items():
            if ready <= now: del ram[k]; return
        ram.popitem(last=False)

    def put(k, ready, l):
        if k in ram: ram.move_to_end(k); return
        while len(ram) >= R: evict_one(l)
        ram[k] = ready

    for t in range(T):
        sc = scored[t]
        tok_stall = 0.0
        for l in range(L):
            m = miss[t, l]
            if far_d: F[keys[t, l]] = np.logaddexp(F[keys[t, l]], t / tau)
            if policy == "today":
                nd = 0
                for k in m:
                    if k in ram: ram.move_to_end(k)
                    else: nd += 1; put(k, 0.0, l)
                st = 0.0 if nd == 0 else a.t_cold + (nd - 1) * a.t_par
                demand += sc * nd
            else:
                wait_until = now
                for k in m:
                    if k in ram:
                        ram.move_to_end(k)
                        if k in pf_set:
                            pf_set.discard(k); pf_hits += sc
                            if ram[k] > now: pf_late += sc
                        wait_until = max(wait_until, ram[k])
                    else:
                        r = ssd_read(); put(k, r, l); wait_until = max(wait_until, r); demand += sc
                st = wait_until - now
            now += st; tok_stall += st
            if victim:
                for k in admitted[t, l]: vram.add(k); ram.pop(k, None)
                for k in evicted[t, l]: vram.discard(k); put(k, now, l)
            else:
                for k in admitted[t, l]: vram.add(k)
                for k in evicted[t, l]: vram.discard(k)
            for ka in k_ahead:
                lt = l + ka
                if lt >= L: continue
                while flight and flight[0] <= now: heapq.heappop(flight)
                for e in A[t, l, ka, :nc]:
                    if len(flight) >= a.qd: break
                    k2 = lt * E + int(e)
                    if k2 in vram or k2 in ram: continue
                    r = ssd_read(); put(k2, r, l); heapq.heappush(flight, r); pf_set.add(k2); pf_reads += sc
            if far_d:
                lt = (l + far_d) % L
                while flight and flight[0] <= now: heapq.heappop(flight)
                n = 0
                for e in np.argsort(-F[lt * E:(lt + 1) * E]):  # the most-used ones not in VRAM/RAM, at any rank
                    if len(flight) >= a.qd or n >= far_n: break
                    k2 = lt * E + int(e)
                    if k2 in vram or k2 in ram: continue
                    r = ssd_read(); put(k2, r, l); heapq.heappush(flight, r); pf_set.add(k2); pf_reads += sc; n += 1
            now += t_layer
        if sc: n_tok += 1; stall_tot += tok_stall
    return (1e3 / (a.t_token + stall_tot / n_tok), demand / n_tok, pf_reads / n_tok, pf_hits / n_tok, pf_late / n_tok)


if A is not None:
    print("Router-ahead predictor: share of layer l+k's 8 experts found in the top-N of its router on layer l's state")
    dec = np.where(scored)[0]
    for k in range(5):
        ls = np.arange(L - k)
        act = keys[dec][:, ls + k] - (ls + k)[None, :, None] * E          # [t, l, 8]
        row = []
        for n in (8, 12, 16):
            pr = A[dec][:, ls, k, :n].astype(np.int64)                     # [t, l, n]
            hit = (act[..., :, None] == pr[..., None, :]).any(-1).mean()
            row.append(f"top-{n} {100 * hit:5.1f}%")
        print(f"  k={k}{' (self-check: must be 100%)' if k == 0 else '':28} " + "  ".join(row))
    print()

print(f"SSD: direct read {a.t_read} ms, {a.t_par} ms/expert in flight, up to {a.qd} prefetches in flight; "
      f"token without SSD waits {a.t_token} ms = {1e3 / a.t_token:.1f} tok/s\n")
print(f"{'RAM for experts':>15} | {'policy':34} | {'tok/s':>6} | {'demand reads/tok':>16} | {'prefetched/tok':>14} | {'used/tok':>8} | {'late/tok':>8}")
# (the k=3 / top-12 / cycle-aware rows of the first study gained nothing: see PLAN.md)
configs = [("today (file cache)", dict(policy="today")), ("own cache + direct reads", dict()),
           ("own cache + direct + victim", dict(victim=True))]
V = dict(victim=True)
configs += [("+victim +far D=5 top-4", dict(V, far_d=5, far_n=4)), ("+victim +far D=15 top-2", dict(V, far_d=15, far_n=2)),
            ("+victim +far D=15 top-4", dict(V, far_d=15, far_n=4)), ("+victim +far D=30 top-4", dict(V, far_d=30, far_n=4))]
if A is not None:
    configs += [("+ahead k=2 top-8 (no victim)", dict(k_ahead=(2,), nc=8)),
                ("+victim +ahead k=2 top-8", dict(V, k_ahead=(2,), nc=8)),
                ("+victim +ahead k=2 +far D=15 top-2", dict(V, k_ahead=(2,), nc=8, far_d=15, far_n=2))]
for gb in (6, 8, 10):
    R = int(gb / EXPERT_GB)
    for name, kw in configs:
        r = run(R, **kw)
        print(f"{gb:8d} GB ({R:5d}) | {name:34} | {r[0]:6.1f} | {r[1]:16.1f} | {r[2]:14.1f} | {r[3]:8.1f} | {r[4]:8.1f}", flush=True)
print(f"\nloading every expert of a layer ahead instead: {L * E * EXPERT_GB:.1f} GB per token from the SSD at "
      f"{EXPERT_GB / a.t_par * 1e3:.1f} GB/s = {L * E * a.t_par / 1e3:.1f} s per token ({1e3 / (L * E * a.t_par):.2f} tok/s)")
print(f"all experts in RAM (24 GB machine): {1e3 / a.t_token:.1f} tok/s")
