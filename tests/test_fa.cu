// Prefill attention (local): attn_fa_k vs attn_prefill_k vs a CPU reference (fp64), then their speed.
//   build\test_fa.exe
// Random int8 K/V with per-position scales, random queries and gates, H = 16, KV = 2, head dim 256, causal from pos0.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "../src/kernels.cuh"

static int g_fail = 0;

struct Case { int pos0, T; };

int main() {
    const int H = 16, KV = 2, ld = H * 2 * AHD;
    const int max_t = 262144 + 64;
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> i8(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f), qd(-2.f, 2.f);

    KVCache kv = kv_alloc(KV, max_t);
    {  // the whole cache once (rows past a case's end are garbage the kernels must ignore: random values)
        std::vector<int8_t> b((size_t)KV * max_t * AHD);
        for (auto& x : b) x = (int8_t)i8(rng);
        CK(cudaMemcpy(kv.k, b.data(), b.size(), cudaMemcpyHostToDevice));
        for (auto& x : b) x = (int8_t)i8(rng);
        CK(cudaMemcpy(kv.v, b.data(), b.size(), cudaMemcpyHostToDevice));
        std::vector<float> s((size_t)KV * max_t);
        for (auto& x : s) x = sc(rng);
        CK(cudaMemcpy(kv.ks, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        for (auto& x : s) x = sc(rng);
        CK(cudaMemcpy(kv.vs, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
    }
    const int TMAX = 2048;
    float *qo, *qg, *o1, *o2;
    int* pos_d;
    CK(cudaMalloc(&qo, (size_t)TMAX * H * AHD * 4)); CK(cudaMalloc(&qg, (size_t)TMAX * ld * 4));
    CK(cudaMalloc(&o1, (size_t)TMAX * H * AHD * 4)); CK(cudaMalloc(&o2, (size_t)TMAX * H * AHD * 4));
    CK(cudaMalloc(&pos_d, 4));
    std::vector<float> hq((size_t)TMAX * H * AHD), hg((size_t)TMAX * ld);
    for (auto& x : hq) x = qd(rng);
    for (auto& x : hg) x = qd(rng);
    CK(cudaMemcpy(qo, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(qg, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice));

    // the CPU reference reads the cache back for the rows it checks
    std::vector<int8_t> hk, hv;
    std::vector<float> hks, hvs;
    auto pull = [&](int n) {
        hk.resize((size_t)KV * n * AHD); hv.resize(hk.size()); hks.resize((size_t)KV * n); hvs.resize(hks.size());
        for (int k = 0; k < KV; k++) {
            CK(cudaMemcpy(hk.data() + (size_t)k * n * AHD, kv.k + (size_t)k * max_t * AHD, (size_t)n * AHD, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data() + (size_t)k * n * AHD, kv.v + (size_t)k * max_t * AHD, (size_t)n * AHD, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hks.data() + (size_t)k * n, kv.ks + (size_t)k * max_t, (size_t)n * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hvs.data() + (size_t)k * n, kv.vs + (size_t)k * max_t, (size_t)n * 4, cudaMemcpyDeviceToHost));
        }
    };
    auto ref_row = [&](int pos0, int t, int h, int n, std::vector<double>& r) {  // out row of token t, head h
        const int kvh = h / (H / KV), p = pos0 + t;
        std::vector<double> s(p + 1);
        double m = -1e300;
        for (int j = 0; j <= p; j++) {
            double d = 0;
            for (int i = 0; i < AHD; i++) d += (double)hq[((size_t)t * H + h) * AHD + i] * hk[((size_t)kvh * n + j) * AHD + i];
            s[j] = d * hks[(size_t)kvh * n + j] / std::sqrt((double)AHD);
            m = std::max(m, s[j]);
        }
        double l = 0;
        r.assign(AHD, 0.0);
        for (int j = 0; j <= p; j++) {
            const double e = std::exp(s[j] - m);
            l += e;
            for (int i = 0; i < AHD; i++) r[i] += e * hv[((size_t)kvh * n + j) * AHD + i] * hvs[(size_t)kvh * n + j];
        }
        for (int i = 0; i < AHD; i++) {
            const double g = hg[(size_t)t * ld + h * 2 * AHD + AHD + i];
            r[i] = r[i] / l / (1.0 + std::exp(-g));
        }
    };

    const Case cases[] = {{0, 37}, {0, 1024}, {5, 3}, {1000, 100}, {4093, 2048}, {30000, 2048}};
    for (const Case& c : cases) {
        CK(cudaMemcpy(pos_d, &c.pos0, 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(o1, 0, (size_t)TMAX * H * AHD * 4)); CK(cudaMemset(o2, 0, (size_t)TMAX * H * AHD * 4));
        attn_prefill(qo, qg, kv, o1, H, KV, pos_d, max_t, ld, c.T, 0);
        attn_fa(qo, qg, kv, o2, H, KV, pos_d, max_t, ld, c.T, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> r1((size_t)c.T * H * AHD), r2(r1.size());
        CK(cudaMemcpy(r1.data(), o1, r1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r2.data(), o2, r2.size() * 4, cudaMemcpyDeviceToHost));
        const int n = c.pos0 + c.T;
        pull(n);
        // reference on a sample of rows: first / last tokens, every head
        double e_new = 0, e_old = 0, mag = 0;
        int bad = 0;
        const int toks[] = {0, c.T / 3, c.T / 2, c.T - 2, c.T - 1};
        std::vector<double> r;
        for (int t : toks) {
            if (t < 0 || t >= c.T) continue;
            for (int h = 0; h < H; h++) {
                ref_row(c.pos0, t, h, n, r);
                for (int i = 0; i < AHD; i++) {
                    const size_t o = ((size_t)t * H + h) * AHD + i;
                    mag = std::max(mag, std::fabs(r[i]));
                    e_new = std::max(e_new, std::fabs(r2[o] - r[i]));
                    e_old = std::max(e_old, std::fabs(r1[o] - r[i]));
                    if (!std::isfinite(r2[o])) bad++;
                }
            }
        }
        double e_ab = 0;  // new vs old over every row
        for (size_t i = 0; i < r1.size(); i++) e_ab = std::max(e_ab, (double)std::fabs(r1[i] - r2[i]));
        const bool ok = bad == 0 && e_new <= 2.0 * e_old + 1e-3 * mag;
        if (!ok) g_fail++;
        printf("pos0 %6d T %5d | vs fp64 ref: new %.2e  old %.2e  (max |out| %.2e) | new vs old %.2e %s\n",
               c.pos0, c.T, e_new, e_old, mag, e_ab, ok ? "ok" : "FAIL");
    }

    // speed: one prefill chunk of 2048 tokens at several context depths
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int pos0 : {0, 16384, 65536, 131072, 196608, 260000}) {
        const int T = 2048;
        if (pos0 + T > max_t) continue;
        CK(cudaMemcpy(pos_d, &pos0, 4, cudaMemcpyHostToDevice));
        float ms[2];
        for (int k = 0; k < 2; k++) {
            auto run = [&] { if (k) attn_fa(qo, qg, kv, o2, H, KV, pos_d, max_t, ld, T, 0); else attn_prefill(qo, qg, kv, o1, H, KV, pos_d, max_t, ld, T, 0); };
            run();
            CK(cudaEventRecord(a));
            const int reps = pos0 >= 65536 ? 2 : 5;
            for (int i = 0; i < reps; i++) run();
            CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
            CK(cudaEventElapsedTime(&ms[k], a, b)); ms[k] /= reps;
        }
        const double flop = 4.0 * H * AHD * T * (pos0 + T / 2.0);  // QK^T and PV, causal
        printf("speed pos0 %6d T %d: old %8.2f ms (%5.1f TFLOPS)  new %8.2f ms (%5.1f TFLOPS)  x%.2f\n",
               pos0, T, ms[0], flop / ms[0] / 1e9, ms[1], flop / ms[1] / 1e9, ms[0] / ms[1]);
    }
    // ---- turbo KV (MOEX_KV=turbo): caches written by qk_norm_rope_k from the same random q/k/v, int8 and turbo ----
    {
        const int LQ = H * 2 * AHD + 2 * KV * AHD, T = 1024, pos0 = 3000, ctx = pos0 + T + 8;
        KVCache c8 = kv_alloc(KV, ctx, 0), ct = kv_alloc(KV, ctx, 1);
        half *qn, *kn;
        CK(cudaMalloc(&qn, AHD * 2)); CK(cudaMemset(qn, 0, AHD * 2));
        CK(cudaMalloc(&kn, AHD * 2)); CK(cudaMemset(kn, 0, AHD * 2));
        float *qkv, *qo2, *oa, *ob, *part;
        const int ns = 4;
        CK(cudaMalloc(&qkv, (size_t)T * LQ * 4)); CK(cudaMalloc(&qo2, (size_t)T * H * AHD * 4));
        CK(cudaMalloc(&oa, (size_t)T * H * AHD * 4)); CK(cudaMalloc(&ob, (size_t)T * H * AHD * 4));
        CK(cudaMalloc(&part, (size_t)T * H * ns * (AHD + 2) * 4));
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> h((size_t)T * LQ);
        auto fill = [&] { for (auto& x : h) x = nd(rng); CK(cudaMemcpy(qkv, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); };
        auto run = [&](KVCache& c, float* out, int p0, int n, int nsplit, float* pt) {
            CK(cudaMemcpy(pos_d, &p0, 4, cudaMemcpyHostToDevice));
            attn_decode(qkv, qkv + H * 2 * AHD, qkv + H * 2 * AHD + KV * AHD, qn, kn, qo2, c, out, H, KV, pos_d, ctx,
                        64, 1e7f, 1e-6f, n, LQ, 0, pt, nsplit);
            CK(cudaDeviceSynchronize());
        };
        for (int p = 0; p < pos0; p += T) {  // the context before the chunk, the same rows in both caches
            const int n = std::min(T, pos0 - p);
            fill(); run(c8, oa, p, n, 1, nullptr); run(ct, ob, p, n, 1, nullptr);
        }
        fill();
        run(c8, oa, pos0, T, 1, nullptr);  // the chunk: FA path, int8
        run(ct, ob, pos0, T, 1, nullptr);  //            FA path, turbo
        std::vector<float> a((size_t)T * H * AHD), b(a.size());
        CK(cudaMemcpy(a.data(), oa, a.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), ob, b.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, cosn = 0, na = 0, nb = 0;
        int bad = 0;
        for (size_t i = 0; i < a.size(); i++) {
            num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)a[i] * a[i];
            cosn += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
            if (!std::isfinite(b[i])) bad++;
        }
        const double rel = std::sqrt(num / den), cs = cosn / std::sqrt(na * nb);
        printf("turbo vs int8 KV, FA path (pos0 %d, T %d): relative RMS %.3f, cosine %.4f%s\n", pos0, T, rel, cs, bad ? " NON-FINITE" : "");
        if (bad || cs < 0.97) g_fail++;
        // the last token again through the decode paths (its K/V rows are rewritten with the same values)
        std::vector<float> last(h.begin() + (size_t)(T - 1) * LQ, h.begin() + (size_t)T * LQ);
        CK(cudaMemcpy(qkv, last.data(), last.size() * 4, cudaMemcpyHostToDevice));
        float *od1, *od2;
        CK(cudaMalloc(&od1, H * AHD * 4)); CK(cudaMalloc(&od2, H * AHD * 4));
        run(ct, od1, pos0 + T - 1, 1, ns, part);  // GQA split + combine
        run(ct, od2, pos0 + T - 1, 1, 1, nullptr);  // attn_decode_k
        std::vector<float> d1(H * AHD), d2(H * AHD);
        CK(cudaMemcpy(d1.data(), od1, d1.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(d2.data(), od2, d2.size() * 4, cudaMemcpyDeviceToHost));
        double e1 = 0, e2 = 0, mg = 0;
        for (int i = 0; i < H * AHD; i++) {
            const float f = b[(size_t)(T - 1) * H * AHD + i];
            e1 = std::max(e1, (double)std::fabs(d1[i] - f)); e2 = std::max(e2, (double)std::fabs(d2[i] - f));
            mg = std::max(mg, (double)std::fabs(f));
        }
        const bool ok = e1 < 0.02 * mg && e2 < 0.02 * mg;
        if (!ok) g_fail++;
        printf("turbo, last token: FA vs GQA decode max diff %.2e, vs attn_decode_k %.2e (max |out| %.2e) %s\n", e1, e2, mg, ok ? "ok" : "FAIL");
    }
    {  // turbo FA speed at depth (garbage cache contents)
        const int T = 2048;
        KVCache ct = kv_alloc(KV, max_t, 1);  // garbage codes, finite norms
        std::vector<float> s((size_t)KV * max_t * 2);
        for (auto& x : s) x = sc(rng) * 50.f;
        CK(cudaMemcpy(ct.ks, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ct.vs, s.data(), s.size() * 4, cudaMemcpyHostToDevice));
        for (int pos0 : {16384, 131072, 260000}) {
            CK(cudaMemcpy(pos_d, &pos0, 4, cudaMemcpyHostToDevice));
            float ms[2];
            for (int k = 0; k < 2; k++) {
                auto go = [&] { attn_fa(qo, k ? nullptr : qg, k ? ct : kv, o2, H, KV, pos_d, max_t, ld, T, 0); };
                go();
                CK(cudaEventRecord(a));
                for (int i = 0; i < 2; i++) go();
                CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
                CK(cudaEventElapsedTime(&ms[k], a, b)); ms[k] /= 2;
            }
            printf("FA speed pos0 %6d T %d: int8 KV %8.2f ms, turbo KV %8.2f ms\n", pos0, T, ms[0], ms[1]);
        }
    }
    printf(g_fail ? "FAILED (%d)\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
