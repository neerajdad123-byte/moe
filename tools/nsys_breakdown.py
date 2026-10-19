# GPU time per kernel, per NVTX range kind (decode / mtp round / prefill), from an Nsight Systems trace of moex.
#   nsys profile -t cuda,nvtx --cuda-graph-trace=node -o prof moex.exe --batch ...
#   nsys export -t sqlite -o prof.sqlite prof.nsys-rep
#   python tools/nsys_breakdown.py prof.sqlite
# A kernel belongs to the range whose host interval holds its launch call (graph nodes: the graph launch).
import bisect, collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
ranges = sorted(db.execute("select start, end, text, globalTid from NVTX_EVENTS where text is not null and end is not null"))
starts = [r[0] for r in ranges]
call = {cid: (s, tid) for s, tid, cid in db.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
names = dict(db.execute("select id, value from StringIds"))

def range_of(cid):
    if cid not in call: return None
    s, tid = call[cid]
    i = bisect.bisect_right(starts, s) - 1
    while i >= 0 and ranges[i][1] < s: i -= 1  # ranges don't nest across kinds; step back over finished ones
    return i if i >= 0 and ranges[i][0] <= s <= ranges[i][1] and ranges[i][3] == tid else None

per = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))  # kind -> label -> [ns, count]
spans = collections.defaultdict(list)  # range index -> kernel/copy intervals
kern = db.execute("select start, end, correlationId, shortName, gridX, gridY, blockX from CUPTI_ACTIVITY_KIND_KERNEL")
for s, e, cid, nm, gx, gy, bx in kern:
    i = range_of(cid)
    if i is None: continue
    kind = ranges[i][2]
    lab = f"{names[nm]} <{gx},{gy}>x{bx}" if names[nm].startswith(("gemv", "gemm", "tc_", "expert")) else names[nm]
    per[kind][lab][0] += e - s; per[kind][lab][1] += 1
    spans[i].append((s, e))
for s, e, cid, b, k in db.execute("select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
    i = range_of(cid)
    if i is None: continue
    lab = f"memcpy kind{k} ({b // 1024} KB)"
    per[ranges[i][2]][lab][0] += e - s; per[ranges[i][2]][lab][1] += 1

for kind in sorted(per):
    idx = [i for i in range(len(ranges)) if ranges[i][2] == kind and spans[i]]  # the trace's tail may miss its GPU events
    n = len(idx)
    wall = sum(ranges[i][1] - ranges[i][0] for i in idx)
    busy = 0
    for i in idx:  # union of GPU intervals launched from this range
        cur = None
        for s, e in sorted(spans[i]):
            if cur and s <= cur[1]: cur[1] = max(cur[1], e)
            else:
                if cur: busy += cur[1] - cur[0]
                cur = [s, e]
        if cur: busy += cur[1] - cur[0]
    tot = sum(v[0] for v in per[kind].values())
    print(f"\n== {kind}: {n} ranges, host wall {wall / n / 1e6:.3f} ms each, GPU busy {busy / n / 1e6:.3f} ms "
          f"({100 * busy / max(wall, 1):.0f}% of wall), kernel sum {tot / n / 1e6:.3f} ms")
    print(f"{'ms/range':>9} {'share':>6} {'calls/range':>11}  kernel <grid>xblock")
    for lab, (t, c) in sorted(per[kind].items(), key=lambda kv: -kv[1][0]):
        if t / tot < 0.002: continue
        print(f"{t / n / 1e6:9.3f} {100 * t / tot:5.1f}% {c / n:11.1f}  {lab}")
