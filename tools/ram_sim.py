"""How fast is moex when the experts don't all fit in RAM (16 GB machines)? Real route traces, no model.

Three tiers: VRAM (the engine's cache: decayed-frequency tau=10, cap slots, bw admissions per layer step), RAM (R
experts), SSD. A VRAM miss runs on the CPU, which needs the expert in RAM; if it isn't, it is read from the SSD on the
critical path. Admission copies of experts not in RAM also read the SSD, in the background.

RAM policies:
  lru      ~ what Windows' file cache does with a memory-mapped model that doesn't fit
  freq     decayed frequency like the VRAM tier, but a slower half-life (RAM changes less often)
  static   lock the R most-used experts (counted on the first half of the prompts, scored on the second half)

  python tools/ram_sim.py build/routes_v2.bin
"""
import argparse
from collections import OrderedDict
import numpy as np

L, E, K = 40, 256, 8
EXPERT_GB = 1.769472e-3
ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--cap", type=int, default=1588)
ap.add_argument("--bw", type=int, default=3)
a = ap.parse_args()

rec = np.fromfile(a.trace, dtype=np.int16).reshape(-1, 12)
T = len(rec) // L
keys = (rec[:, 2].astype(np.int64)[:, None] * E + rec[:, 4:].astype(np.int64)).reshape(T, L, K)
prompt = rec[::L, 0].astype(int)
decode = rec[::L, 3] == 1
half = prompt.max() // 2
train = prompt <= half
counts = np.bincount(keys[train].ravel(), minlength=L * E)


def run(R, policy, tau_ram=200.0):
    slot_of = np.full(L * E, -1, np.int64)
    skey = np.full(a.cap, -1, np.int64)
    sscore = np.full(a.cap, -np.inf)
    G = np.full(L * E, -np.inf)
    ram = OrderedDict()                        # lru: key -> None, oldest first
    GR = np.full(L * E, -np.inf)
    inr = np.zeros(L * E, bool)
    static = set(np.argsort(-counts)[:R].tolist()) if policy == "static" else None
    crit = bg = n_tok = cpu_exp = 0

    def in_ram(k, t):
        """Is expert k in RAM when needed at token t? Updates the RAM tier (a read from SSD inserts it)."""
        if policy == "static": return k in static
        if policy == "lru":
            if k in ram: ram.move_to_end(k); return True
            ram[k] = None
            if len(ram) > R: ram.popitem(last=False)
            return False
        GR[k] = np.logaddexp(GR[k], t / tau_ram)
        if inr[k]: return True
        if inr.sum() >= R:  # evict the lowest decayed frequency
            inr[np.argmin(np.where(inr, GR, np.inf))] = False
        inr[k] = True
        return False

    for t in range(T):
        score = not train[t] and decode[t]
        for l in range(L):
            ks = keys[t, l]
            G[ks] = np.logaddexp(G[ks], t / 10.0)
            sl = slot_of[ks]
            h = sl >= 0
            for s_ in sl[h]: sscore[s_] = G[skey[s_]]
            for k in ks[~h]:                    # CPU misses: need RAM
                hit = in_ram(int(k), t)
                if score: crit += not hit; cpu_exp += 1
            protect = sl[h].tolist()
            budget = a.bw
            for k in ks[~h]:                    # admissions copy RAM -> VRAM (already touched above)
                if budget == 0: break
                sc = sscore.copy(); sc[protect] = np.inf
                v = int(np.argmin(sc))
                if skey[v] >= 0: slot_of[skey[v]] = -1
                skey[v] = k; slot_of[k] = v; sscore[v] = G[k]; protect.append(v); budget -= 1
        n_tok += score
    return crit / n_tok, cpu_exp / n_tok


print(f"{T} tokens; scored: decode tokens of prompts {half + 1}-{prompt.max()} (static ranks from prompts 0-{half})")
print("SSD reads on the critical path per token (CPU misses whose expert isn't in RAM)\n")
print(f"{'RAM for experts':>16} {'experts':>8} | {'lru (OS cache)':>14} {'freq tau200':>12} {'static top-R':>13}")
for gb in (4, 6, 8, 10, 12, 14, 18.2):
    R = min(int(gb / EXPERT_GB), L * E)
    row = [run(R, p)[0] for p in ("lru", "freq", "static")]
    print(f"{gb:13.1f} GB {R:8d} | {row[0]:14.1f} {row[1]:12.1f} {row[2]:13.1f}", flush=True)
