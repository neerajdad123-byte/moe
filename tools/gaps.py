"""GPU idle-gap analysis of an nsys cuda_gpu_trace CSV: where does the GPU wait between operations?

  python tools/gaps.py build/prof_c_cuda_gpu_trace.csv
Attributes each gap to the operation that ends it (what the GPU was waiting to start).
"""
import csv, sys
from collections import defaultdict

rows = []
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f):
        start = float(r["Start (ns)"]); dur = float(r["Duration (ns)"] or 0)
        rows.append((start, start + dur, r["Name"]))
rows.sort()
# steady state only: skip the first 40% (load, warm-up, cold prompt)
rows = rows[int(len(rows) * 0.4):]
t0, t1 = rows[0][0], max(e for _, e, _ in rows)
busy = gap_total = 0.0
gap_by = defaultdict(lambda: [0.0, 0])
end = rows[0][0]
for s, e, n in rows:
    if s > end:
        g = s - end
        gap_total += g
        key = n.split("(")[0].replace("void ", "").split("<")[0][:40]
        gap_by[key][0] += g; gap_by[key][1] += 1
    busy += max(0.0, e - max(s, end))
    end = max(end, e)
span = t1 - t0
print(f"span {span / 1e6:.0f} ms: GPU busy {100 * busy / span:.1f}%, idle {100 * gap_total / span:.1f}%")
print("idle time by the operation that ended the wait:")
for k, (g, c) in sorted(gap_by.items(), key=lambda kv: -kv[1][0])[:12]:
    print(f"  {k:40s} {g / 1e6:8.1f} ms  {100 * g / span:5.1f}%  ({c} gaps, avg {g / c / 1e3:.1f} us)")
