#!/usr/bin/env python3
"""Offline cache-policy simulator over a real MoEx route trace.

Replays build/bench50/route_trace.jsonl (579,936 lines from the 50-prompt
all-Python run) chronologically through several candidate expert-caching
policies at a fixed capacity (matching the real GPU arena, 1112 slots), and
reports each policy's hit rate. Every "online" policy here only uses
information available up to that point in the stream (no future peeking);
the Belady oracle is the one deliberate exception, kept as an upper-bound
reference, never a deployable policy.

Cache key = (layer, expert) -- matches the real DeviceModel: capacity is one
global budget shared across all layers, not per-layer.

Usage: python tools/simulate_policies.py [trace_path] [capacity]
"""
import sys
import json
import time
import heapq
from collections import OrderedDict, defaultdict

TRACE_PATH = sys.argv[1] if len(sys.argv) > 1 else "build/bench50/route_trace.jsonl"
CAPACITY = int(sys.argv[2]) if len(sys.argv) > 2 else 1112
N_LAYERS = 48
N_EXPERTS = 128


def load_events(path):
    """One event per route_trace.jsonl line: (layer, tuple(8 expert ids))."""
    events = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            events.append((d["layer"], tuple(d["experts"])))
    return events


# ---------------------------------------------------------------------------
# Policies. Each is a generator-free function: given `events` and `capacity`,
# returns (hits, misses, evictions).
# ---------------------------------------------------------------------------

def sim_lru(events, capacity):
    cache = OrderedDict()  # key -> True, MRU at end
    hits = misses = evictions = 0
    for layer, experts in events:
        for e in experts:
            k = (layer, e)
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                    evictions += 1
                cache[k] = True
    return hits, misses, evictions


def sim_lfu_global(events, capacity):
    freq = defaultdict(int)
    resident = set()
    heap = []  # (freq_at_push, key); lazy deletion via freq check
    hits = misses = evictions = 0
    for layer, experts in events:
        for e in experts:
            k = (layer, e)
            freq[k] += 1
            if k in resident:
                hits += 1
                heapq.heappush(heap, (freq[k], k))
            else:
                misses += 1
                if len(resident) >= capacity:
                    while True:
                        f, victim = heapq.heappop(heap)
                        if victim in resident and f == freq[victim]:
                            resident.discard(victim)
                            evictions += 1
                            break
                resident.add(k)
                heapq.heappush(heap, (freq[k], k))
    return hits, misses, evictions


def sim_ema_hotness(events, capacity, decay=0.001):
    """EMA hotness: score updates only on touch (score = score*(1-decay)+1),
    evict lowest-score resident. Adaptive to recent shifts without a hard
    periodic refresh (checklist B/§windowed frequency). Lazy-heap with a
    version stamp per key (not float equality) for correct/robust eviction."""
    score = defaultdict(float)
    resident = set()
    heap = []  # (score, key, stamp)
    latest_stamp = defaultdict(int)
    hits = misses = evictions = 0
    for layer, experts in events:
        for e in experts:
            k = (layer, e)
            score[k] = score[k] * (1 - decay) + 1.0
            if k in resident:
                hits += 1
            else:
                misses += 1
                if len(resident) >= capacity:
                    while True:
                        s, victim, stamp = heapq.heappop(heap)
                        if victim in resident and stamp == latest_stamp[victim]:
                            resident.discard(victim)
                            evictions += 1
                            break
                resident.add(k)
            latest_stamp[k] += 1
            heapq.heappush(heap, (score[k], k, latest_stamp[k]))
    return hits, misses, evictions


def sim_pin_periodic(events, capacity, pin_top_k, pin_every_tokens):
    """Replicates the real C++ policy: cumulative per-layer frequency core,
    refreshed every `pin_every_tokens` *tokens* (a "token" = one full
    48-layer pass; the real run refreshed every 5 *prompts*, ~= every
    5*avg_tokens_per_prompt tokens -- parameterized here per-token so it can
    be swept finely), rest is plain LRU. Pure frequency, no co-occurrence
    (see sim_cluster_core_mates for that variant)."""
    cache = OrderedDict()
    pinned = set()
    hist = defaultdict(int)  # cumulative (layer,expert) -> count, layer-scoped ranking
    hits = misses = evictions = 0
    token_count = 0
    cur_layer_seen = -1

    def refresh_pins():
        pinned.clear()
        by_layer = defaultdict(list)
        for (l, e), c in hist.items():
            by_layer[l].append((c, e))
        for l, lst in by_layer.items():
            lst.sort(reverse=True)
            core = [e for _, e in lst[:pin_top_k]]
            for e in core:
                pinned.add((l, e))

    for layer, experts in events:
        if layer <= cur_layer_seen:
            token_count += 1
            if token_count % pin_every_tokens == 0:
                refresh_pins()
        cur_layer_seen = layer
        for e in experts:
            k = (layer, e)
            hist[k] += 1
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                if len(cache) >= capacity:
                    # evict LRU among unpinned
                    victim = None
                    for cand_k in cache:
                        if cand_k not in pinned:
                            victim = cand_k
                            break
                    if victim is None:
                        victim = next(iter(cache))
                    del cache[victim]
                    evictions += 1
                cache[k] = True
    return hits, misses, evictions


def sim_cluster_core_mates(events, capacity, pin_top_k, n_mates, pin_every_tokens):
    """Core (top-K/layer by cumulative freq) + N co-occurrence mates (experts
    that frequently appear alongside the core in the SAME layer/token, even if
    not individually top-K -- the "mid-frequency but locked together" idea),
    refreshed periodically. Co-occurrence learned online (only past data)."""
    cache = OrderedDict()
    pinned = set()
    hist = defaultdict(int)
    co = defaultdict(lambda: defaultdict(int))  # (layer,e1) -> {e2: count}
    hits = misses = evictions = 0
    token_count = 0
    cur_layer_seen = -1

    def refresh_pins():
        pinned.clear()
        by_layer = defaultdict(list)
        for (l, e), c in hist.items():
            by_layer[l].append((c, e))
        for l, lst in by_layer.items():
            lst.sort(reverse=True)
            core = [e for _, e in lst[:pin_top_k]]
            for e in core:
                pinned.add((l, e))
            if n_mates > 0 and core:
                mate_score = defaultdict(int)
                core_set = set(core)
                for c_e in core:
                    for mate, w in co[(l, c_e)].items():
                        if mate not in core_set:
                            mate_score[mate] += w
                top_mates = sorted(mate_score.items(), key=lambda x: -x[1])[:n_mates]
                for mate, _ in top_mates:
                    pinned.add((l, mate))

    for layer, experts in events:
        if layer <= cur_layer_seen:
            token_count += 1
            if token_count % pin_every_tokens == 0:
                refresh_pins()
        cur_layer_seen = layer
        for e in experts:
            hist[(layer, e)] += 1
        for e1 in experts:
            for e2 in experts:
                if e1 != e2:
                    co[(layer, e1)][e2] += 1
        for e in experts:
            k = (layer, e)
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                if len(cache) >= capacity:
                    victim = None
                    for cand_k in cache:
                        if cand_k not in pinned:
                            victim = cand_k
                            break
                    if victim is None:
                        victim = next(iter(cache))
                    del cache[victim]
                    evictions += 1
                cache[k] = True
    return hits, misses, evictions


def sim_hybrid_lru_lfu(events, capacity, w_recency=0.5, w_freq=0.5):
    """Checklist A "hybrid LRU/LFU": blend recency rank and frequency rank
    into one score, evict the global minimum -- soft blend, no hard pin
    partition (contrast with the pin-based policies, which hard-protect a
    core and let everything else be pure LRU)."""
    freq = defaultdict(int)
    last_seen = defaultdict(int)
    resident = set()
    heap = []  # (score, key, stamp)
    latest_stamp = defaultdict(int)
    hits = misses = evictions = 0
    tick = 0
    for layer, experts in events:
        for e in experts:
            tick += 1
            k = (layer, e)
            freq[k] += 1
            last_seen[k] = tick
            if k in resident:
                hits += 1
            else:
                misses += 1
                if len(resident) >= capacity:
                    while True:
                        s, victim, stamp = heapq.heappop(heap)
                        if victim in resident and stamp == latest_stamp[victim]:
                            resident.discard(victim)
                            evictions += 1
                            break
                resident.add(k)
            # lower score = more evictable. Recency term: how stale (tick -
            # last_seen), normalized by tick; frequency term: -log(freq).
            # Heap pops the MIN as the eviction victim, so items that SHOULD
            # be kept need a HIGH score: reward freshness (low staleness) and
            # high frequency, both positively.
            staleness = (tick - last_seen[k]) / max(1, tick)
            score = -w_recency * staleness + w_freq * (freq[k] ** 0.5)
            latest_stamp[k] += 1
            heapq.heappush(heap, (score, k, latest_stamp[k]))
    return hits, misses, evictions


def sim_lift_cluster_mates(events, capacity, pin_top_k, n_mates, pin_every_tokens,
                            min_freq_for_lift=3):
    """Same core+mates shape as sim_cluster_core_mates, but mates are ranked
    by LIFT = P(i,j) / (P(i)*P(j)) instead of raw co-occurrence count
    (checklist F "lift-based co-occurrence"). Raw co-occurrence favors pairs
    that are simply both globally hot; lift favors pairs that appear together
    MORE than their individual popularity would predict by chance -- the
    "locked together" signal, isolated from "both happen to be popular"."""
    cache = OrderedDict()
    pinned = set()
    hist = defaultdict(int)
    co = defaultdict(lambda: defaultdict(int))
    total_touches_per_layer = defaultdict(int)
    hits = misses = evictions = 0
    token_count = 0
    cur_layer_seen = -1

    def refresh_pins():
        pinned.clear()
        by_layer = defaultdict(list)
        for (l, e), c in hist.items():
            by_layer[l].append((c, e))
        for l, lst in by_layer.items():
            lst.sort(reverse=True)
            core = [e for _, e in lst[:pin_top_k]]
            for e in core:
                pinned.add((l, e))
            if n_mates > 0 and core:
                n_l = max(1, total_touches_per_layer[l])
                mate_score = defaultdict(float)
                core_set = set(core)
                for c_e in core:
                    p_c = hist[(l, c_e)] / n_l
                    if hist[(l, c_e)] < min_freq_for_lift:
                        continue
                    for mate, w in co[(l, c_e)].items():
                        if mate in core_set:
                            continue
                        p_m = hist[(l, mate)] / n_l
                        if p_m <= 0 or hist[(l, mate)] < min_freq_for_lift:
                            continue
                        p_joint = w / n_l
                        lift = p_joint / (p_c * p_m)
                        mate_score[mate] = max(mate_score[mate], lift)
                top_mates = sorted(mate_score.items(), key=lambda x: -x[1])[:n_mates]
                for mate, _ in top_mates:
                    pinned.add((l, mate))

    for layer, experts in events:
        if layer <= cur_layer_seen:
            token_count += 1
            if token_count % pin_every_tokens == 0:
                refresh_pins()
        cur_layer_seen = layer
        total_touches_per_layer[layer] += 1
        for e in experts:
            hist[(layer, e)] += 1
        for e1 in experts:
            for e2 in experts:
                if e1 != e2:
                    co[(layer, e1)][e2] += 1
        for e in experts:
            k = (layer, e)
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                if len(cache) >= capacity:
                    victim = None
                    for cand_k in cache:
                        if cand_k not in pinned:
                            victim = cand_k
                            break
                    if victim is None:
                        victim = next(iter(cache))
                    del cache[victim]
                    evictions += 1
                cache[k] = True
    return hits, misses, evictions


def sim_jaccard_cluster(events, capacity, budget_per_layer, theta, pin_every_tokens):
    """Checklist H "route-pattern clustering": greedy Jaccard-similarity
    cluster growth per layer (support(e) = set of tokens where e appeared;
    Jaccard(i,j) = |support(i) & support(j)| / |support(i) | support(j)|,
    approximated online via co-occurrence/frequency counts without storing
    full support sets). Seed from the hottest unclustered expert, grow while
    average Jaccard to current members >= theta, cap cluster size. Pin
    whole clusters (highest-total-frequency clusters first) up to
    budget_per_layer/layer -- this is the direct "predict the repeating
    cluster, not the single expert" mechanism, distinct from ranking
    individual experts by hotness (sim_pin_periodic) or by pairwise lift
    (sim_lift_cluster_mates)."""
    cache = OrderedDict()
    pinned = set()
    hist = defaultdict(int)
    co = defaultdict(lambda: defaultdict(int))
    hits = misses = evictions = 0
    token_count = 0
    cur_layer_seen = -1

    def jaccard(l, i, j):
        cij = co[(l, i)].get(j, 0)
        denom = hist[(l, i)] + hist[(l, j)] - cij
        return cij / denom if denom > 0 else 0.0

    def refresh_pins():
        pinned.clear()
        by_layer = defaultdict(list)
        for (l, e), c in hist.items():
            by_layer[l].append((c, e))
        for l, lst in by_layer.items():
            lst.sort(reverse=True)
            # Bound the candidate pool and cluster size -- an unbounded
            # O(candidates^2 * cluster_size) greedy grow over up to 128
            # experts/layer is billions of ops per refresh; this is only ever
            # called ~10 times total (once per pin_every window) but still
            # needs to stay in the tens-of-millions-of-ops range to finish in
            # reasonable time. Top-40/layer by frequency is still a superset
            # of anything a 12-40 slot pin budget could use anyway.
            candidates = [e for _, e in lst[:40] if hist[(l, e)] > 0]
            max_cluster = min(12, max(6, budget_per_layer // 2))
            clustered = set()
            clusters = []
            for seed in candidates:
                if seed in clustered:
                    continue
                cluster = [seed]
                clustered.add(seed)
                while len(cluster) < max_cluster:
                    best_cand, best_sim = None, theta
                    for cand in candidates:
                        if cand in clustered:
                            continue
                        avg_sim = sum(jaccard(l, cand, m) for m in cluster) / len(cluster)
                        if avg_sim >= best_sim:
                            best_sim = avg_sim
                            best_cand = cand
                    if best_cand is None:
                        break
                    cluster.append(best_cand)
                    clustered.add(best_cand)
                total_freq = sum(hist[(l, e)] for e in cluster)
                clusters.append((total_freq, cluster))
            clusters.sort(key=lambda x: -x[0])
            budget = budget_per_layer
            for _, cluster in clusters:
                if budget <= 0:
                    break
                for e in cluster:
                    if budget <= 0:
                        break
                    if (l, e) not in pinned:
                        pinned.add((l, e))
                        budget -= 1

    for layer, experts in events:
        if layer <= cur_layer_seen:
            token_count += 1
            if token_count % pin_every_tokens == 0:
                refresh_pins()
        cur_layer_seen = layer
        for e in experts:
            hist[(layer, e)] += 1
        for e1 in experts:
            for e2 in experts:
                if e1 != e2:
                    co[(layer, e1)][e2] += 1
        for e in experts:
            k = (layer, e)
            if k in cache:
                cache.move_to_end(k)
                hits += 1
            else:
                misses += 1
                if len(cache) >= capacity:
                    victim = None
                    for cand_k in cache:
                        if cand_k not in pinned:
                            victim = cand_k
                            break
                    if victim is None:
                        victim = next(iter(cache))
                    del cache[victim]
                    evictions += 1
                cache[k] = True
    return hits, misses, evictions


def sim_belady_oracle(events, capacity):
    """Future-informed optimal (Belady's MIN algorithm): always evict the
    resident key whose NEXT use is furthest away (or never again). Upper
    bound reference only -- not an achievable online policy.

    Efficient lazy-heap form: a resident key's "next future occurrence"
    doesn't change until that key is touched again, so push a fresh
    (-next_occ, key, stamp) entry only on each touch (hit or miss) instead of
    rescanning all resident keys on every eviction -- O(N log capacity)
    instead of O(evictions * capacity), which is the difference between
    minutes and not finishing."""
    # IMPORTANT: indices must be on the FLAT (per-key-touch) scale, not the
    # outer per-event (per-8-key-line) scale -- an earlier version recorded
    # the outer event index here while the replay loop below indexes into
    # `flat` (8x as many positions), silently comparing two different
    # scales. That made every "next use" estimate ~8x too soon, so the
    # "oracle" evicted based on garbage distances and scored far below
    # plain LRU -- mathematically impossible for a correct Belady
    # implementation, caught by a stress test against LRU before trusting it.
    next_occurrence = defaultdict(list)
    flat = []
    flat_idx = 0
    for layer, experts in events:
        for e in experts:
            k = (layer, e)
            flat.append(k)
            next_occurrence[k].append(flat_idx)
            flat_idx += 1
    key_next_ptr = defaultdict(int)

    def next_use_after(k, cur_idx):
        occs = next_occurrence[k]
        p = key_next_ptr[k]
        while p < len(occs) and occs[p] <= cur_idx:
            p += 1
        key_next_ptr[k] = p
        return occs[p] if p < len(occs) else float("inf")

    resident = set()
    heap = []  # (-next_occ, key, stamp)
    latest_stamp = defaultdict(int)
    hits = misses = evictions = 0

    for idx, k in enumerate(flat):
        if k in resident:
            hits += 1
        else:
            misses += 1
            if len(resident) >= capacity:
                while True:
                    neg_next, victim, stamp = heapq.heappop(heap)
                    if victim in resident and stamp == latest_stamp[victim]:
                        resident.discard(victim)
                        evictions += 1
                        break
            resident.add(k)
        nxt = next_use_after(k, idx)
        latest_stamp[k] += 1
        heapq.heappush(heap, (-nxt, k, latest_stamp[k]))
    return hits, misses, evictions


def pct(h, m):
    return 100.0 * h / (h + m) if (h + m) else 0.0


def main():
    t0 = time.time()
    print(f"loading {TRACE_PATH} ...")
    events = load_events(TRACE_PATH)
    print(f"  {len(events)} layer-touches, {len(events)*8} key-touches, "
          f"loaded in {time.time()-t0:.1f}s")
    print(f"capacity = {CAPACITY} (matches real GPU arena)\n")

    results = []

    def run(name, fn, *args, **kwargs):
        t = time.time()
        h, m, ev = fn(events, CAPACITY, *args, **kwargs)
        dt = time.time() - t
        r = {"name": name, "hits": h, "misses": m, "evictions": ev,
             "hit_pct": pct(h, m), "seconds": round(dt, 1)}
        results.append(r)
        print(f"  {name:34s} hit {r['hit_pct']:6.2f}%   evictions {ev:9d}   ({dt:.1f}s)")
        return r

    print("=== A. Baselines (individual-expert hotness, no cluster awareness) ===")
    run("lru_no_pin", sim_lru)
    run("lfu_global", sim_lfu_global)
    run("ema_hotness_decay1e-3", sim_ema_hotness, 0.001)
    run("ema_hotness_decay1e-4", sim_ema_hotness, 0.0001)
    run("hybrid_lru_lfu", sim_hybrid_lru_lfu, 0.5, 0.5)

    print("\n=== B. Real-C++-policy replica (validation) ===")
    # real run: pin top-17/layer, refresh every 5 prompts. Avg tokens/prompt
    # in the real run ~= 4639488/8/48 events... approximate via token count:
    # 50 prompts, most at 200 decode + ~20-40 prefill -> ~ (579936/48) tokens total
    total_tokens = len(events) // N_LAYERS
    tokens_per_prompt = total_tokens / 50.0
    every_tokens = max(1, round(tokens_per_prompt * 5))
    print(f"  (total_tokens={total_tokens}, ~{tokens_per_prompt:.1f}/prompt, "
        f"pin_every_tokens~={every_tokens} approximates 'every 5 prompts')")
    run("replica_pin17_every5prompts", sim_pin_periodic, 17, every_tokens)

    print("\n=== C. Pin hyperparameter sweep (frequency-core only -- 'hot experts') ===")
    for k in (12, 17, 27, 40):
        for every_p in (1, 5, 10):
            et = max(1, round(tokens_per_prompt * every_p))
            run(f"pin_k{k}_every{every_p}p [HOT]", sim_pin_periodic, k, et)

    print("\n=== D. Cluster core + raw co-occurrence mates ('travels with the core') ===")
    for k in (12, 17, 27):
        for nm in (5, 15):
            et = max(1, round(tokens_per_prompt * 5))
            run(f"cluster_core{k}_mates{nm}_every5p [CLUSTER]",
                sim_cluster_core_mates, k, nm, et)

    print("\n=== E. Cluster core + LIFT-normalized mates (filters out pairs that")
    print("     only look 'together' because both are independently hot) ===")
    for k in (12, 17, 27):
        for nm in (5, 15):
            et = max(1, round(tokens_per_prompt * 5))
            run(f"lift_core{k}_mates{nm}_every5p [CLUSTER]",
                sim_lift_cluster_mates, k, nm, et)

    print("\n=== F. Jaccard-similarity greedy clustering (checklist H --")
    print("     pin whole discovered clusters, not individually-ranked experts) ===")
    for budget in (17, 27, 40):
        for theta in (0.15, 0.25):
            et = max(1, round(tokens_per_prompt * 5))
            run(f"jaccard_budget{budget}_theta{theta}_every5p [CLUSTER]",
                sim_jaccard_cluster, budget, theta, et)

    print("\n=== G. Oracle upper bound (NOT achievable online -- future-informed) ===")
    run("belady_oracle_upper_bound [ORACLE]", sim_belady_oracle)

    results.sort(key=lambda r: -r["hit_pct"])
    print("\n=== RANKED (excluding oracle) ===")
    for r in results:
        if "oracle" not in r["name"]:
            print(f"  {r['hit_pct']:6.2f}%  {r['name']}")

    hot_best = max((r for r in results if "[HOT]" in r["name"] or r["name"] in
                    ("lru_no_pin", "lfu_global", "ema_hotness_decay1e-3",
                     "ema_hotness_decay1e-4", "hybrid_lru_lfu")),
                   key=lambda r: r["hit_pct"], default=None)
    cluster_best = max((r for r in results if "[CLUSTER]" in r["name"]),
                       key=lambda r: r["hit_pct"], default=None)
    if hot_best and cluster_best:
        print(f"\n=== HOT-EXPERT best vs CLUSTER-AWARE best (the core question) ===")
        print(f"  best individual-hotness policy: {hot_best['hit_pct']:.2f}%  ({hot_best['name']})")
        print(f"  best cluster-aware policy:      {cluster_best['hit_pct']:.2f}%  ({cluster_best['name']})")
        print(f"  delta: {cluster_best['hit_pct'] - hot_best['hit_pct']:+.2f} percentage points")

    with open("build/bench50/policy_sim_results.json", "w") as f:
        json.dump({"capacity": CAPACITY, "trace": TRACE_PATH,
                   "total_events": len(events), "results": results}, f, indent=2)
    print("\nwrote build/bench50/policy_sim_results.json")
    print(f"total simulation time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
