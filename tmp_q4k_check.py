import numpy as np, gguf

def rd_h(p, i):
    return int(p[i]) | (int(p[i+1]) << 8)

def fp16_to_fp32(h):
    h = int(h)
    sign = (h & 0x8000) << 16
    exp = (h >> 10) & 0x1f
    mant = h & 0x3ff
    if exp == 0:
        if mant == 0:
            b = sign
        else:
            e = -1; m = mant
            while (m & 0x400) == 0:
                m <<= 1; e += 1
            m &= 0x3ff
            b = sign | ((127 - 15 - e) << 23) | (m << 13)
    elif exp == 0x1f:
        b = sign | 0x7f800000 | (mant << 13)
    else:
        b = sign | ((exp + 112) << 23) | (mant << 13)
    return np.frombuffer(np.uint32(b).tobytes(), dtype=np.float32)[0]

def gsm(j, q):
    q = q.astype(np.int64)
    if j < 4:
        d = q[j] & 63; m = q[j + 4] & 63
    else:
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m

def deq_full(p):
    d = fp16_to_fp32(rd_h(p, 0)); dmin = fp16_to_fp32(rd_h(p, 2))
    scales = p[4:16].astype(np.int64)
    qs = p[16:144].astype(np.int64)
    y = np.zeros(256, dtype=np.float64); is_ = 0; qo = 0
    for j in range(0, 256, 64):
        s1, m1 = gsm(is_ + 0, scales); s2, m2 = gsm(is_ + 1, scales)
        dd1 = d * s1; mm1 = dmin * m1; dd2 = d * s2; mm2 = dmin * m2
        for l in range(32):
            y[l + qo] = dd1 * (qs[l] & 0xF) - mm1
            y[l + qo + 32] = dd2 * (qs[l] >> 4) - mm2
        qo += 64; is_ += 2
    return y.astype(np.float32)

r = gguf.GGUFReader('C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf')
t = next(x for x in r.tensors if x.name == 'blk.25.ffn_up_exps.weight')
blk = 144
for e in [20, 92, 104]:
    raw = np.ascontiguousarray(t.data[e])
    my = np.concatenate([deq_full(raw[b*blk:(b+1)*blk]) for b in range(raw.size//blk)])
    ref = gguf.quants.dequantize(raw, t.tensor_type)
    print('e%d: my_nan=%d ref_nan=%d maxabs=%.3e' % (
        e, int((~np.isfinite(my)).sum()), int((~np.isfinite(ref)).sum()),
        float(np.nanmax(np.abs(my - ref)))))
