"""Cold SSD read speed for experts, bypassing the Windows file cache (FILE_FLAG_NO_BUFFERING). No model run.

Reads random experts (1.77 MB each, 4 KB-aligned superset) from the container files: one at a time (the latency a
CPU miss waits) and with N reads in flight (the throughput a background prefetcher gets).

  python tools/ssd_bench.py C:/models/qwen36_i4_gs64
"""
import ctypes, ctypes.wintypes as wt, json, mmap, os, random, struct, sys, time
from concurrent.futures import ThreadPoolExecutor

root = sys.argv[1] if len(sys.argv) > 1 else "C:/models/qwen36_i4_gs64"
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
k32.SetFilePointerEx.argtypes = [wt.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]
GENERIC_READ, SHARE, OPEN_EXISTING, NOBUF = 0x80000000, 1, 3, 0x20000000

# expert blobs: (file, offset, bytes) of merged_weight + qs, from the safetensors headers
experts = []
for fn in sorted(os.listdir(root)):
    if not fn.endswith(".safetensors"): continue
    with open(os.path.join(root, fn), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
    blobs = {}
    for k, v in h.items():
        if ".mlp.experts." in k and k.startswith("model.layers."):
            blobs.setdefault(k.rsplit(".", 1)[0], []).append(8 + n + v["data_offsets"][0])
            blobs[k.rsplit(".", 1)[0]].append(8 + n + v["data_offsets"][1])
    for parts in blobs.values():  # [a, b] per tensor: merged_weight and qs are not adjacent in the file
        experts.append((os.path.join(root, fn), [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]))
print(f"{len(experts)} experts indexed")

def read_one(e):
    path, ranges = e
    h = k32.CreateFileW(path, GENERIC_READ, SHARE, None, OPEN_EXISTING, NOBUF, None)
    total = 0
    for a, b in ranges:
        a0, b0 = a & ~4095, (b + 4095) & ~4095
        buf = mmap.mmap(-1, b0 - a0)  # page-aligned
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        k32.SetFilePointerEx(h, a0, None, 0)
        got = wt.DWORD()
        ok = k32.ReadFile(h, addr, b0 - a0, ctypes.byref(got), None)
        del addr; buf.close()
        if not ok or got.value != b0 - a0: raise OSError(ctypes.get_last_error())
        total += b - a
    k32.CloseHandle(h)
    assert total == 1769472, total
    return total

random.seed(1)
sample = random.sample(experts, 600)
t0 = time.perf_counter(); lat = []
for e in sample[:120]:
    s = time.perf_counter(); read_one(e); lat.append(time.perf_counter() - s)
lat.sort()
print(f"one at a time: median {1e3 * lat[len(lat) // 2]:.2f} ms/expert, p90 {1e3 * lat[int(len(lat) * .9)]:.2f} ms "
      f"-> {1.769472 / (sum(lat) / len(lat)) / 1e3:.2f} GB/s")
for q in (2, 4, 8, 16):
    batch = sample[120 + (q - 2) * 30: 120 + (q - 2) * 30 + 30 * q // 2] if False else random.sample(experts, 120)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(q) as ex: total = sum(ex.map(read_one, batch))
    dt = time.perf_counter() - t0
    print(f"{q:2d} in flight: {total / dt / 1e9:.2f} GB/s = {len(batch) / dt:.0f} experts/s ({1e3 * dt / len(batch):.2f} ms/expert)")

POOL = ThreadPoolExecutor(32)  # alive across reads, like the engine's reader threads
# one expert split into P parallel chunk reads (what a demand read could do): latency per expert
def read_chunks(e, P):
    path, ranges = e
    jobs = []
    for a, b in ranges:
        a0, b0 = a & ~4095, (b + 4095) & ~4095
        step = max(4096, ((b0 - a0) // P + 4095) & ~4095)
        jobs += [(path, [(x, min(x + step, b0))]) for x in range(a0, b0, step)]
    def one(j):
        path, [(x, y)] = j
        h = k32.CreateFileW(path, GENERIC_READ, SHARE, None, OPEN_EXISTING, NOBUF, None)
        buf = mmap.mmap(-1, y - x); addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        k32.SetFilePointerEx(h, x, None, 0); got = wt.DWORD()
        ok = k32.ReadFile(h, addr, y - x, ctypes.byref(got), None)
        del addr; buf.close(); k32.CloseHandle(h)
        assert ok and got.value == y - x
    list(POOL.map(one, jobs))

for P in (1, 4, 8, 16):
    lat = []
    for e in random.sample(experts, 80):
        s = time.perf_counter(); read_chunks(e, P); lat.append(time.perf_counter() - s)
    lat.sort()
    print(f"one expert as {P:2d} parallel chunks: median {1e3 * lat[len(lat) // 2]:.2f} ms, p90 {1e3 * lat[int(len(lat) * .9)]:.2f} ms")
