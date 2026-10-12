"""Tensor quant types of a GGUF file (local path or HF URL; for a URL only the header is fetched).

  python tools/gguf_types.py <path-or-url>
"""
import struct, sys, collections, urllib.request

src = sys.argv[1]
if src.startswith("http"):
    req = urllib.request.Request(src, headers={"Range": "bytes=0-33554431"})  # header fits in 32 MB
    data = urllib.request.urlopen(req).read()
else:
    data = open(src, "rb").read(33554432)

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 39: "MXFP4"}
pos = 0
def rd(fmt):
    global pos
    v = struct.unpack_from("<" + fmt, data, pos); pos += struct.calcsize("<" + fmt); return v[0]
def rstr():
    global pos
    n = rd("Q"); s = data[pos:pos + n].decode("utf-8", "replace"); pos += n; return s
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        at, n = rd("I"), rd("Q")
        return [rval(at) for _ in range(n)] if n < 64 else [rval(at) for _ in range(n)][:0]
    return rd(SCALAR[t])

assert data[:4] == b"GGUF", "not a GGUF file"
pos = 4
ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
kv = {}
for _ in range(nkv):
    k = rstr(); kv[k] = rval(rd("I"))
print(f"GGUF v{ver}, {nt} tensors; file_type={kv.get('general.file_type')}")
by_kind = collections.defaultdict(collections.Counter)
for _ in range(nt):
    name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q")
    kind = ".".join(p for p in name.split(".") if not p.isdigit())
    by_kind[kind][TYPES.get(t, str(t))] += 1
for kind in sorted(by_kind):
    print(f"  {kind:40s} {dict(by_kind[kind])}")
