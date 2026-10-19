// Kernel correctness (vs ref/golden from qwen36_ref.py) + bandwidth bench.
//   build\test_kernels.exe [filter]
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "../src/st.h"
#include "../src/kernels.cuh"
#include "../src/cpu_expert.h"
#include <chrono>

static const char* MODEL = getenv("QWEN36_MODEL") ? getenv("QWEN36_MODEL") : "C:\\models\\qwen36_i4_gs64";
static const float EPS = 1e-6f;
static int g_fail = 0;

static std::vector<float> golden(const std::string& n) {
    std::string p = "ref/golden/" + n + ".bin";
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing golden %s (run ref/qwen36_ref.py --dump)\n", p.c_str()); exit(1); }
    _fseeki64(f, 0, SEEK_END); size_t b = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<float> v(b / 4);
    fread(v.data(), 4, v.size(), f); fclose(f);
    return v;
}

template <class T> static T* up(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static float* dalloc(size_t n, bool zero = false) {
    float* d; CK(cudaMalloc(&d, n * 4)); if (zero) CK(cudaMemset(d, 0, n * 4)); return d;
}
static std::vector<float> down(const float* d, size_t n) {
    std::vector<float> v(n); CK(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost)); return v;
}
static half* wup(const Container& c, const std::string& n) {
    auto b = c.read(n);
    half* d; CK(cudaMalloc(&d, b.size())); CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice));
    return d;
}

// Error relative to the reference's largest magnitude: robust to near-zero elements.
static void check(const char* what, const std::vector<float>& got, const std::vector<float>& ref, float tol) {
    double mx = 0, err = 0;
    for (size_t i = 0; i < ref.size(); i++) { mx = std::max(mx, (double)fabsf(ref[i])); err = std::max(err, (double)fabsf(got[i] - ref[i])); }
    double rel = err / (mx + 1e-30);
    bool ok = got.size() == ref.size() && rel < tol && std::isfinite(rel);
    printf("  %-34s max_abs_err %.3e  rel %.3e  %s\n", what, err, rel, ok ? "PASS" : "FAIL");
    if (!ok) g_fail++;
}

// Median of `reps` timed runs, after warm-up, with L2 flushed before each (decode never
// re-reads a layer's weights from L2, so a warm-L2 number would lie). Returns ms.
static float bench(const char* what, double bytes, const std::function<void()>& fn, int reps = 50) {
    static void* flush = nullptr; const size_t FL = 64 << 20;  // > 24 MB L2
    if (!flush) CK(cudaMalloc(&flush, FL));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; i++) fn();
    std::vector<float> t(reps);
    for (int i = 0; i < reps; i++) {
        CK(cudaMemsetAsync(flush, i, FL));
        cudaEventRecord(a); fn(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&t[i], a, b);
    }
    std::sort(t.begin(), t.end());
    float ms = t[reps / 2];
    printf("  %-34s %8.4f ms  %7.1f GB/s  (%4.1f%% of 188)\n", what, ms, bytes / ms / 1e6, bytes / ms / 1e6 / 188 * 100);
    return ms;
}

static bool want(const char* f, const char* name) { return !f || strstr(name, f); }
static std::string pn_(int p, const char* rest) { char b[64]; snprintf(b, sizeof b, "p%03d_%s", p, rest); return b; }

int main(int argc, char** argv) {
    const char* filt = argc > 1 ? argv[1] : nullptr;
    Container C(MODEL);
    const int D = 2048;
    const std::string P = "p005_";

    if (want(filt, "K1")) {
        printf("K1 rmsnorm\n");
        half* post0 = wup(C, "model.layers.0.post_attention_layernorm.weight");
        half* post3 = wup(C, "model.layers.3.post_attention_layernorm.weight");
        half* in0 = wup(C, "model.layers.0.input_layernorm.weight");
        float* x = up(golden(P + "L00_x_in")); float* d = up(golden(P + "L00_dn_out")); float* y = dalloc(D);
        rmsnorm(x, d, post0, y, D, EPS);
        check("L00 residual+post_ln -> moe_in", down(y, D), golden(P + "L00_moe_in"), 1e-5f);
        float* x3 = up(golden(P + "L03_x_in")); float* d3 = up(golden(P + "L03_attn_out"));
        rmsnorm(x3, d3, post3, y, D, EPS);
        check("L03 residual+post_ln -> moe_in", down(y, D), golden(P + "L03_moe_in"), 1e-5f);
        bench("rmsnorm 2048", 3.0 * D * 4, [&] { rmsnorm(x, nullptr, in0, y, D, EPS); });
    }

    if (want(filt, "K2")) {
        printf("K2 gemv_f16\n");
        half* in0 = wup(C, "model.layers.0.input_layernorm.weight");
        half* Wqkv = wup(C, "model.layers.0.linear_attn.in_proj_qkv.weight");
        half* Wout = wup(C, "model.layers.0.linear_attn.out_proj.weight");
        float* x = up(golden(P + "L00_x_in")); float* h = dalloc(D); float* y = dalloc(8192);
        rmsnorm(x, nullptr, in0, h, D, EPS);
        gemv_f16(Wqkv, h, y, D, 8192);
        check("L00 in_proj_qkv -> dn_qkv", down(y, 8192), golden(P + "L00_dn_qkv"), 1e-5f);
        auto o = golden(P + "L00_dn_o");  // [32,128] gated-norm output = out_proj input
        float* ov = up(o);
        gemv_f16(Wout, ov, y, 4096, D);
        check("L00 out_proj -> dn_out", down(y, D), golden(P + "L00_dn_out"), 1e-5f);
        bench("qkv 8192x2048", 8192.0 * D * 2, [&] { gemv_f16(Wqkv, h, y, D, 8192); });
        bench("out_proj 2048x4096", 2048.0 * 4096 * 2, [&] { gemv_f16(Wout, ov, y, 4096, D); });
        CK(cudaFree(Wqkv)); CK(cudaFree(Wout));

        half* lm = wup(C, "lm_head.weight"); half* nw = wup(C, "model.norm.weight");
        const int V = 248320;
        float* xf = up(golden(P + "x_final")); float* logits = dalloc(V);
        rmsnorm(xf, nullptr, nw, h, D, EPS);
        gemv_f16(lm, h, logits, D, V);
        auto got = down(logits, V), ref = golden(P + "logits");
        check("lm_head -> logits", got, ref, 1e-5f);
        int ag = int(std::max_element(got.begin(), got.end()) - got.begin());
        int ar = int(std::max_element(ref.begin(), ref.end()) - ref.begin());
        printf("  argmax got %d ref %d %s\n", ag, ar, ag == ar ? "PASS" : "FAIL"); if (ag != ar) g_fail++;
        bench("lm_head 248320x2048", (double)V * D * 2, [&] { gemv_f16(lm, h, logits, D, V); }, 20);
    }

    if (want(filt, "K3") || want(filt, "K4") || want(filt, "K5")) {
        printf("K3 router + K4 experts + K5 shared expert\n");
        const int E = 256, K = 8, SI = 512;
        float *logits = dalloc(E), *w = dalloc(K), *act = dalloc(K * EI), *out = dalloc(D), *scr = dalloc(2 * SI + D + 1);
        int* idx; CK(cudaMalloc(&idx, K * 4));
        uint8_t* arena; CK(cudaMalloc(&arena, K * EXPERT_BYTES));
        const uint8_t** slots; CK(cudaMalloc(&slots, K * sizeof(void*)));
        for (int L : {0, 3, 39}) {
            char pf[32]; snprintf(pf, sizeof pf, "%sL%02d_", P.c_str(), L);
            std::string lp = "model.layers." + std::to_string(L) + ".mlp.";
            half* Wr = wup(C, lp + "gate.weight");
            float* h = up(golden(pf + std::string("moe_in")));
            gemv_f16(Wr, h, logits, D, E);
            check((std::string(pf) + "router logits").c_str(), down(logits, E), golden(pf + std::string("router_logits")), 1e-5f);
            router_topk(logits, E, K, true, idx, w);
            std::vector<int> gi(K); CK(cudaMemcpy(gi.data(), idx, K * 4, cudaMemcpyDeviceToHost));
            auto ri = golden(pf + std::string("route_idx"));  // int32 bits read as float
            bool same = !memcmp(gi.data(), ri.data(), K * 4);
            printf("  %-34s %s\n", (std::string(pf) + "top-8 ids").c_str(), same ? "PASS" : "FAIL"); if (!same) g_fail++;
            check((std::string(pf) + "top-8 weights").c_str(), down(w, K), golden(pf + std::string("route_w")), 1e-5f);
            // load the selected experts into the arena (test stand-in for the slot cache)
            std::vector<const uint8_t*> hs(K);
            for (int k = 0; k < K; k++) {
                std::string ep = lp + "experts." + std::to_string(gi[k]) + ".";
                auto q = C.read(ep + "merged_weight"), s = C.read(ep + "qs");
                uint8_t* dst = arena + k * EXPERT_BYTES;
                CK(cudaMemcpy(dst, q.data(), q.size(), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dst + EXPERT_Q_BYTES, s.data(), s.size(), cudaMemcpyHostToDevice));
                hs[k] = dst;
            }
            CK(cudaMemcpy(slots, hs.data(), K * sizeof(void*), cudaMemcpyHostToDevice));
            half *Wg = wup(C, lp + "shared_expert.gate_proj.weight"), *Wu = wup(C, lp + "shared_expert.up_proj.weight"),
                 *Wd = wup(C, lp + "shared_expert.down_proj.weight"), *Ws = wup(C, lp + "shared_expert_gate.weight");
            half* Wgu; CK(cudaMalloc(&Wgu, (size_t)2 * SI * D * 2));  // fused [gate; up], as the engine stores it
            CK(cudaMemcpy(Wgu, Wg, (size_t)SI * D * 2, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(Wgu + (size_t)SI * D, Wu, (size_t)SI * D * 2, cudaMemcpyDeviceToDevice));
            experts_forward(slots, w, K, h, act, out);
            shared_expert_add({Wgu}, {Wd}, Ws, h, out, scr, D, SI);
            check((std::string(pf) + "routed+shared -> moe_out").c_str(), down(out, D), golden(pf + std::string("moe_out")), 1e-5f);
            if (L == 0) {
                bench("router gemv 256x2048 + topk", 256.0 * D * 2, [&] { gemv_f16(Wr, h, logits, D, E); router_topk(logits, E, K, true, idx, w); });
                bench("experts x8 int4 gs64", (double)K * EXPERT_BYTES, [&] { experts_forward(slots, w, K, h, act, out); });
                const double gu_b = K * (EXPERT_Q_BYTES * 2.0 / 3 + 2.0 * EI * (D / 64) * 4), dn_b = K * (EXPERT_Q_BYTES / 3.0 + D * (EI / 64) * 4.0);
                bench("  gate_up only", gu_b, [&] { expert_gate_up_k<8><<<persistent_blocks(), 256>>>(slots, K, nullptr, h, act); });
                bench("  down only", dn_b, [&] { expert_down_k<8><<<persistent_blocks(), 256, K * xpad_size(EI) * 4>>>(slots, w, K, nullptr, act, out); });
                bench("shared expert F16", 3.0 * SI * D * 2, [&] { shared_expert_add({Wgu}, {Wd}, Ws, h, out, scr, D, SI); });
            }
            for (void* p : {(void*)Wr, (void*)h, (void*)Wg, (void*)Wu, (void*)Wgu, (void*)Wd, (void*)Ws}) CK(cudaFree(p));
        }
    }

    if (want(filt, "POOL")) {
        printf("CpuPool stress: 100k jobs, each item must run exactly once\n");
        CpuPool pool(12);
        std::vector<std::atomic<int>> hit(64);
        long long bad = 0;
        for (int job = 0; job < 100000; job++) {
            int items = 1 + job % 64;
            for (int i = 0; i < items; i++) hit[i] = 0;
            std::function<void(int)> f = [&](int i) { hit[i].fetch_add(1); };
            pool.parallel_for(items, f);
            for (int i = 0; i < items; i++) bad += hit[i] != 1;
        }
        printf("  bad items %lld %s\n", bad, bad ? "FAIL" : "PASS"); if (bad) g_fail++;
    }

    if (want(filt, "K14")) {
        printf("K14 CPU experts (AVX2) vs GPU K4, L00 routed experts\n");
        const int K = 8;
        std::string lp = "model.layers.0.mlp.";
        auto h = golden(P + "L00_moe_in");
        auto ri = golden(P + "L00_route_idx"); auto rwv = golden(P + "L00_route_w");
        std::vector<int> ids(K); memcpy(ids.data(), ri.data(), K * 4);
        std::vector<std::vector<uint8_t>> blobs(K);
        std::vector<const uint8_t*> hq(K); std::vector<const float*> hs(K), dslot(K);
        uint8_t* arena; CK(cudaMalloc(&arena, K * EXPERT_BYTES));
        for (int k = 0; k < K; k++) {
            std::string ep = lp + "experts." + std::to_string(ids[k]) + ".";
            auto q = C.read(ep + "merged_weight"), s = C.read(ep + "qs");
            blobs[k] = q; blobs[k].insert(blobs[k].end(), s.begin(), s.end());  // same layout as a VRAM slot
            hq[k] = blobs[k].data(); hs[k] = (const float*)(blobs[k].data() + EXPERT_Q_BYTES);
            CK(cudaMemcpy(arena + k * EXPERT_BYTES, blobs[k].data(), EXPERT_BYTES, cudaMemcpyHostToDevice));
        }
        std::vector<const uint8_t*> ds(K); for (int k = 0; k < K; k++) ds[k] = arena + k * EXPERT_BYTES;
        const uint8_t** slots; CK(cudaMalloc(&slots, K * sizeof(void*))); CK(cudaMemcpy(slots, ds.data(), K * sizeof(void*), cudaMemcpyHostToDevice));
        float *hd = up(h), *wd = up(rwv), *act = dalloc(K * EI), *out = dalloc(D);
        experts_forward(slots, wd, K, hd, act, out);
        auto gpu = down(out, D);
        for (int threads : {1, 2, 4, 8, 12}) {
            CpuExperts ce(threads);
            std::vector<float> cpu(D);
            ce.run(K, hq.data(), hs.data(), rwv.data(), h.data(), cpu.data());
            if (threads == 8) check("CPU 8 experts vs GPU", cpu, gpu, 1e-5f);
            for (int m : {1, 2, 4, 8}) {
                for (int i = 0; i < 3; i++) ce.run(m, hq.data(), hs.data(), rwv.data(), h.data(), cpu.data());
                std::vector<double> t(21);
                static std::vector<uint8_t> l3(64 << 20);  // > 24 MB L3: misses are read from DRAM, not cache
                for (auto& ti : t) {
                    for (size_t i = 0; i < l3.size(); i += 64) l3[i]++;
                    auto a = std::chrono::steady_clock::now();
                    ce.run(m, hq.data(), hs.data(), rwv.data(), h.data(), cpu.data());
                    ti = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
                }
                std::sort(t.begin(), t.end());
                printf("  threads %2d  %d expert(s)  %7.3f ms  %5.1f GB/s  (%.3f ms/expert)\n", threads, m, t[10],
                       m * EXPERT_BYTES / t[10] / 1e6, t[10] / m);
            }
        }
        CK(cudaFree(arena));
        // Same kernel on mmapped experts: first touch in this process vs again (page-fault cost).
        CpuExperts ce(8);
        std::vector<float> cpu(D);
        for (int L : {5, 6}) {
            std::string lpm = "model.layers." + std::to_string(L) + ".mlp.experts.";
            for (int pass = 0; pass < 2; pass++) {
                double tot = 0;
                for (int e0 = 0; e0 < 64; e0 += 2) {
                    const uint8_t* mq[2]; const float* ms[2];
                    for (int j = 0; j < 2; j++) {
                        mq[j] = C.map(lpm + std::to_string(e0 + j) + ".merged_weight");
                        ms[j] = (const float*)C.map(lpm + std::to_string(e0 + j) + ".qs");
                    }
                    auto a = std::chrono::steady_clock::now();
                    ce.run(2, mq, ms, rwv.data(), h.data(), cpu.data());
                    tot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
                }
                printf("  mmap L%02d pass %d (%s): %.3f ms/expert over 64 experts\n", L, pass, pass ? "touched" : "first touch", tot / 64);
            }
        }
    }

    if (want(filt, "K10")) {
        printf("K10 gemv_i8 (per-row int8, quantized on GPU)\n");
        half* in0 = wup(C, "model.layers.0.input_layernorm.weight");
        float* x = up(golden(P + "L00_x_in")); float* h = dalloc(D);
        rmsnorm(x, nullptr, in0, h, D, EPS);
        auto hh = down(h, D);
        struct Case { const char* name; int O, I; } cases[] = {
            {"model.layers.0.linear_attn.in_proj_qkv.weight", 8192, D},
            {"model.layers.0.linear_attn.out_proj.weight", D, 4096},
            {"lm_head.weight", 248320, D}};
        for (auto& cs : cases) {
            half* W = wup(C, cs.name);
            int8_t* q; float* s; CK(cudaMalloc(&q, (size_t)cs.O * cs.I)); CK(cudaMalloc(&s, cs.O * 4));
            quantize_rows_i8_k<<<cs.O, 256>>>(W, q, s, cs.I);
            std::vector<float> xin(cs.I);
            for (int i = 0; i < cs.I; i++) xin[i] = hh[i % D] * (1 + (i / D));  // reuse a real activation for I=4096
            float* xd = up(xin); float* y = dalloc(cs.O); float* yf = dalloc(cs.O);
            gemv({nullptr, q, s}, xd, y, cs.I, cs.O);
            gemv_f16(W, xd, yf, cs.I, cs.O);
            // exact int8 reference on the host, first 4096 rows
            int R = std::min(cs.O, 4096);
            std::vector<int8_t> hq((size_t)R * cs.I); std::vector<float> hs(R);
            CK(cudaMemcpy(hq.data(), q, hq.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs.data(), s, R * 4, cudaMemcpyDeviceToHost));
            std::vector<float> ref(R);
            for (int r = 0; r < R; r++) { double a = 0; for (int i = 0; i < cs.I; i++) a += (double)hq[(size_t)r * cs.I + i] * xin[i]; ref[r] = float(a * hs[r]); }
            auto got = down(y, cs.O); got.resize(R);
            std::string nm = std::string(cs.name).substr(std::string(cs.name).rfind('.', strlen(cs.name) - 8) + 1);
            check((nm + " kernel vs exact int8").c_str(), got, ref, 1e-5f);
            // quantization error vs F16 (informational: this is model drift, not a kernel bug)
            auto gq = down(y, cs.O), gf = down(yf, cs.O);
            double mx = 0, e = 0; for (int i = 0; i < cs.O; i++) { mx = std::max(mx, (double)fabsf(gf[i])); e = std::max(e, (double)fabsf(gq[i] - gf[i])); }
            printf("  %-34s int8-vs-F16 rel %.3e (quantization drift)\n", nm.c_str(), e / mx);
            bench((nm + " int8").c_str(), (double)cs.O * cs.I + cs.O * 4.0, [&] { gemv({nullptr, q, s}, xd, y, cs.I, cs.O); }, cs.O > 100000 ? 20 : 50);
            for (void* p : {(void*)W, (void*)q, (void*)s, (void*)xd, (void*)y, (void*)yf}) CK(cudaFree(p));
        }
    }

    if (want(filt, "K11")) {
        printf("K11 gemv_q4 (int4 gs64 dense) vs exact int4 math; GEMM int4 path vs GEMV\n");
        struct Case { const char* name; int O, I; } cases[] = {
            {"model.layers.0.linear_attn.in_proj_qkv.weight", 8192, D}, {"model.layers.0.linear_attn.out_proj.weight", D, 4096},
            {"model.layers.0.mlp.shared_expert.down_proj.weight", D, 512}, {"lm_head.weight", 248320, D}};
        auto hh = golden(P + "L00_moe_in");
        for (auto& cs : cases) {
            half* W = wup(C, cs.name);
            uint8_t* q; float* s;
            CK(cudaMalloc(&q, (size_t)cs.O * cs.I / 2)); CK(cudaMalloc(&s, (size_t)cs.O * (cs.I / 64) * 4));
            quantize_rows_q4_k<<<cs.O, 64>>>(W, q, s, cs.I);
            std::vector<float> xin(cs.I); for (int i = 0; i < cs.I; i++) xin[i] = hh[i % D] * ((i / D) ? -1.f : 1.f);
            float *xd = up(xin), *y = dalloc(cs.O);
            Mat m; m.q4 = q; m.s = s;
            gemv(m, xd, y, cs.I, cs.O);
            int R = std::min(cs.O, 2048);
            std::vector<uint8_t> hq((size_t)R * cs.I / 2); std::vector<float> hs((size_t)R * (cs.I / 64));
            CK(cudaMemcpy(hq.data(), q, hq.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hs.data(), s, hs.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<float> ref(R);
            for (int r = 0; r < R; r++) {
                double a = 0;
                for (int i = 0; i < cs.I; i++) {
                    int b = hq[((size_t)r * cs.I + i) / 2], v = (i & 1) ? b >> 4 : b & 0xF;
                    a += (double)(v >= 8 ? v - 16 : v) * hs[(size_t)r * (cs.I / 64) + i / 64] * xin[i];
                }
                ref[r] = (float)a;
            }
            auto got = down(y, cs.O); got.resize(R);
            std::string nm = std::string(cs.name).substr(std::string(cs.name).rfind('.', strlen(cs.name) - 8) + 1);
            check((nm + " gemv_q4 vs exact").c_str(), got, ref, 1e-5f);
            bench((nm + " int4").c_str(), (double)cs.O * cs.I / 2 + cs.O * (cs.I / 64) * 4.0, [&] { gemv(m, xd, y, cs.I, cs.O); }, cs.O > 100000 ? 20 : 50);
            if (cs.O < 100000) {  // prefill GEMM on int4 weights vs the GEMV, 16 tokens
                const int S = 16;
                std::vector<float> hx((size_t)S * cs.I); for (int t = 0; t < S; t++) for (int i = 0; i < cs.I; i++) hx[(size_t)t * cs.I + i] = xin[i] * (1.f + 0.03f * t);
                float *X = up(hx), *Y = dalloc((size_t)S * cs.O), *Yr = dalloc((size_t)S * cs.O);
                gemm(m, X, Y, S, cs.I, cs.O);
                for (int t = 0; t < S; t++) gemv(m, X + (size_t)t * cs.I, Yr + (size_t)t * cs.O, cs.I, cs.O);
                check((nm + " gemm int4 vs gemv").c_str(), down(Y, (size_t)S * cs.O), down(Yr, (size_t)S * cs.O), 5e-3f);
                for (void* p : {(void*)X, (void*)Y, (void*)Yr}) CK(cudaFree(p));
            }
            for (void* p : {(void*)W, (void*)q, (void*)s, (void*)xd, (void*)y}) CK(cudaFree(p));
        }
    }

    if (want(filt, "PF1")) {
        printf("PF1 prefill MoE on tensor cores vs pairs_forward (layer 0 experts, T=256 rows, top-8 of NE)\n");
        const int T = 256, K = 8;
        for (int NE : {256, 32}) {  // 256: ~8 rows per expert (a real chunk); 32: ~64 rows per expert
            uint8_t* arena; CK(cudaMalloc(&arena, (size_t)NE * EXPERT_BYTES));
            for (int e = 0; e < NE; e++) {
                std::string ep = "model.layers.0.mlp.experts." + std::to_string(e) + ".";
                auto q = C.read(ep + "merged_weight"), s = C.read(ep + "qs");
                CK(cudaMemcpy(arena + (size_t)e * EXPERT_BYTES, q.data(), q.size(), cudaMemcpyHostToDevice));
                CK(cudaMemcpy(arena + (size_t)e * EXPERT_BYTES + EXPERT_Q_BYTES, s.data(), s.size(), cudaMemcpyHostToDevice));
            }
            std::vector<float> hx((size_t)T * D);  // real activations (12 golden positions), varied per row
            for (int t = 0; t < T; t++) {
                auto g = golden(pn_(t % 12, "L00_moe_in"));
                for (int i = 0; i < D; i++) hx[(size_t)t * D + i] = g[i] * (1.f + 0.01f * (t / 12));
            }
            // routing: K distinct experts per row (deterministic), grouped by expert as the engine does
            std::vector<std::vector<Pair>> per(NE);
            uint32_t rng = 12345;
            for (int t = 0; t < T; t++) {
                std::vector<int> pick;
                while ((int)pick.size() < K) {
                    rng = rng * 1664525u + 1013904223u;
                    int e = (rng >> 8) % NE;
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                for (int r = 0; r < K; r++) per[pick[r]].push_back({t, t * K + r, 0.05f + 0.1f * r});
            }
            std::vector<PairItem> items; std::vector<Pair> pairs;
            for (int e = 0; e < NE; e++) if (!per[e].empty()) {
                items.push_back({arena + (size_t)e * EXPERT_BYTES, (int)pairs.size(), (int)per[e].size()});
                pairs.insert(pairs.end(), per[e].begin(), per[e].end());
            }
            PairItem* items_d; Pair* pairs_d;
            CK(cudaMalloc(&items_d, items.size() * sizeof(PairItem))); CK(cudaMalloc(&pairs_d, pairs.size() * sizeof(Pair)));
            CK(cudaMemcpy(items_d, items.data(), items.size() * sizeof(PairItem), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(pairs_d, pairs.data(), pairs.size() * sizeof(Pair), cudaMemcpyHostToDevice));
            float *X = up(hx), *act = dalloc((size_t)T * K * EI, true), *yr = dalloc((size_t)T * K * D, true),
                  *act2 = dalloc((size_t)T * K * EI, true), *yt = dalloc((size_t)T * K * D, true);
            const int ni = (int)items.size();
            pairs_forward(items_d, ni, pairs_d, X, act, yr);
            pairs_forward_tc(items_d, ni, pairs_d, X, act2, yt);
            char lbl[96];
            snprintf(lbl, sizeof lbl, "NE=%d act (silu(g)*u)", NE);
            check(lbl, down(act2, (size_t)T * K * EI), down(act, (size_t)T * K * EI), 5e-3f);  // fp16 operands
            snprintf(lbl, sizeof lbl, "NE=%d ypair", NE);
            check(lbl, down(yt, (size_t)T * K * D), down(yr, (size_t)T * K * D), 5e-3f);
            const double bytes = (double)ni * EXPERT_BYTES;
            snprintf(lbl, sizeof lbl, "  NE=%d pairs_forward (GEMV)", NE);
            float ms_old = bench(lbl, bytes, [&] { pairs_forward(items_d, ni, pairs_d, X, act, yr); }, 10);
            snprintf(lbl, sizeof lbl, "  NE=%d pairs_forward_tc", NE);
            float ms_new = bench(lbl, bytes, [&] { pairs_forward_tc(items_d, ni, pairs_d, X, act2, yt); }, 10);
            printf("      -> %.2fx, %.1f TFLOPS\n", ms_old / ms_new, (double)T * K * 3 * EI * D * 2 / (ms_new * 1e9));
            for (void* p : {(void*)arena, (void*)items_d, (void*)pairs_d, (void*)X, (void*)act, (void*)yr, (void*)act2, (void*)yt})
                CK(cudaFree(p));
        }
    }

    if (want(filt, "K12")) {
        printf("K12 prefill GEMM (WMMA fp16, int8 weights) vs per-token GEMV\n");
        struct Case { const char* name; int O, I; } cases[] = {
            {"model.layers.0.linear_attn.in_proj_qkv.weight", 8192, D}, {"model.layers.0.linear_attn.out_proj.weight", D, 4096}};
        for (auto& cs : cases) {
            half* W = wup(C, cs.name);
            int8_t* q; float* s; CK(cudaMalloc(&q, (size_t)cs.O * cs.I)); CK(cudaMalloc(&s, cs.O * 4));
            quantize_rows_i8_k<<<cs.O, 256>>>(W, q, s, cs.I);
            Mat m; m.q = q; m.s = s;
            const int SMAX = 1024;
            std::vector<float> hx((size_t)SMAX * cs.I);
            for (int t = 0; t < SMAX; t++) {  // real activations (12 golden positions), varied per token
                auto g = golden(pn_(t % 12, "L00_moe_in"));
                for (int i = 0; i < cs.I; i++) hx[(size_t)t * cs.I + i] = g[i % D] * (1.f + 0.01f * (t / 12)) * ((i / D) ? -1.f : 1.f);
            }
            float *X = up(hx), *Y = dalloc((size_t)SMAX * cs.O), *Yr = dalloc((size_t)SMAX * cs.O);
            for (int S : {64, 128, 256, 1024}) {
                gemm(m, X, Y, S, cs.I, cs.O);
                for (int t = 0; t < S; t++) gemv(m, X + (size_t)t * cs.I, Yr + (size_t)t * cs.O, cs.I, cs.O);
                auto g = down(Y, (size_t)S * cs.O), r = down(Yr, (size_t)S * cs.O);
                std::string nm = std::string(cs.name).substr(std::string(cs.name).rfind('.', strlen(cs.name) - 8) + 1);
                char lbl[96]; snprintf(lbl, sizeof lbl, "%s S=%d vs GEMV", nm.c_str(), S);
                check(lbl, g, r, 5e-3f);  // fp16 activations: ~1e-3 relative, below the int8 weight error
                snprintf(lbl, sizeof lbl, "  S=%d gemm (%.3f ms/token)", S, 0.0);
                float ms = bench(lbl, (double)cs.O * cs.I, [&] { gemm(m, X, Y, S, cs.I, cs.O); }, 20);
                printf("      -> %.4f ms per token (GEMV per token: weights re-read S times)\n", ms / S);
            }
            for (void* p : {(void*)W, (void*)q, (void*)s, (void*)X, (void*)Y, (void*)Yr}) CK(cudaFree(p));
        }
    }

    if (want(filt, "K8T")) {
        printf("K8T DeltaNet over a chunk (L00, all 12 golden positions in one call)\n");
        const int C8 = 8192, VH = 32, KH = 16, T = 12, LDR = C8 + VH * DNV + 2 * VH;
        std::string lp = "model.layers.0.", la = lp + "linear_attn.";
        half *ln = wup(C, lp + "input_layernorm.weight"), *Wqkv = wup(C, la + "in_proj_qkv.weight"),
             *Wz = wup(C, la + "in_proj_z.weight"), *Wa = wup(C, la + "in_proj_a.weight"), *Wb = wup(C, la + "in_proj_b.weight"),
             *Wo = wup(C, la + "out_proj.weight"), *cw = wup(C, la + "conv1d.weight"), *Al = wup(C, la + "A_log"),
             *dtb = wup(C, la + "dt_bias"), *nw = wup(C, la + "norm.weight");
        float *h = dalloc(D), *rows = dalloc((size_t)T * LDR), *conv = dalloc((size_t)T * C8), *o = dalloc((size_t)T * VH * DNV), *out = dalloc(D);
        float *ring = dalloc(C8 * 3, true), *S = dalloc((size_t)VH * DNK * DNV, true);
        for (int p = 0; p < T; p++) {  // fused projection row per token: [qkv | z | a | b]
            float* x = up(golden(pn_(p, "L00_x_in")));
            rmsnorm(x, nullptr, ln, h, D, EPS);
            float* r = rows + (size_t)p * LDR;
            gemv_f16(Wqkv, h, r, D, C8); gemv_f16(Wz, h, r + C8, D, VH * DNV);
            gemv_f16(Wa, h, r + C8 + VH * DNV, D, VH); gemv_f16(Wb, h, r + C8 + VH * DNV + VH, D, VH);
            CK(cudaFree(x));
        }
        deltanet_core(rows, rows + C8, rows + C8 + VH * DNV, rows + C8 + VH * DNV + VH, LDR, cw, Al, dtb, nw, ring, S, conv, o,
                      C8, 4, VH, KH, EPS, T);
        double worst = 0;
        for (int p = 0; p < T; p++) {
            gemv_f16(Wo, o + (size_t)p * VH * DNV, out, VH * DNV, D);
            auto g = down(out, D), r = golden(pn_(p, "L00_dn_out"));
            double mx = 0, e = 0; for (int i = 0; i < D; i++) { mx = std::max(mx, (double)fabsf(r[i])); e = std::max(e, (double)fabsf(g[i] - r[i])); }
            worst = std::max(worst, e / mx);
        }
        printf("  worst rel dn_out over 12 positions (one chunk) %.3e %s\n", worst, worst < 1e-4 ? "PASS" : "FAIL");
        if (worst >= 1e-4) g_fail++;
        bench("deltanet core, T=12 chunk", 2.0 * VH * DNK * DNV * 4, [&] {
            deltanet_core(rows, rows + C8, rows + C8 + VH * DNV, rows + C8 + VH * DNV + VH, LDR, cw, Al, dtb, nw, ring, S, conv, o, C8, 4, VH, KH, EPS, T); });
    }

    if (want(filt, "MEMCPY")) {
        printf("MEMCPY: staging copy speed, mmapped experts -> pinned host buffer\n");
        const int NB = 24;
        uint8_t* pin; CK(cudaMallocHost(&pin, (size_t)NB * EXPERT_BYTES));
        std::vector<const uint8_t*> q(256 * 4), sc(256 * 4);
        for (int l = 4; l < 8; l++) for (int e = 0; e < 256; e++) {
            std::string ep = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
            q[(l - 4) * 256 + e] = C.map(ep + "merged_weight"); sc[(l - 4) * 256 + e] = C.map(ep + "qs");
        }
        for (int pass = 0; pass < 2; pass++)
            for (int th : {1, 2, 4, 8}) {
                CpuPool pool(th);
                auto t0 = std::chrono::steady_clock::now();
                for (int b = 0; b < 1024 / NB; b++) {
                    std::function<void(int)> f = [&](int i) {
                        int k = b * NB + i;
                        memcpy(pin + (size_t)i * EXPERT_BYTES, q[k], EXPERT_Q_BYTES);
                        memcpy(pin + (size_t)i * EXPERT_BYTES + EXPERT_Q_BYTES, sc[k], EXPERT_BYTES - EXPERT_Q_BYTES);
                    };
                    pool.parallel_for(NB, f);
                }
                double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                printf("  pass %d, %d thread(s): %.2f GB/s\n", pass, th, (1024 / NB) * NB * EXPERT_BYTES / s / 1e9);
            }
        // the alternatives in the engine's situation: with a spinning 8-thread pool alongside, and the driver's
        // own pageable copy (no staging buffer of ours)
        {
            CpuPool spin(8), pool(4);
            auto t0 = std::chrono::steady_clock::now();
            for (int b = 0; b < 1024 / NB; b++) {
                std::function<void(int)> f = [&](int i) {
                    int k = b * NB + i;
                    memcpy(pin + (size_t)i * EXPERT_BYTES, q[k], EXPERT_Q_BYTES);
                    memcpy(pin + (size_t)i * EXPERT_BYTES + EXPERT_Q_BYTES, sc[k], EXPERT_BYTES - EXPERT_Q_BYTES);
                };
                pool.parallel_for(NB, f);
            }
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("  4 copy threads next to an idle-spinning 8-thread pool: %.2f GB/s\n", (1024 / NB) * NB * EXPERT_BYTES / s / 1e9);
        }
        uint8_t* dst; CK(cudaMalloc(&dst, (size_t)NB * EXPERT_BYTES));
        for (int pass = 0; pass < 2; pass++) {
            auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < 1024; k++) {
                uint8_t* d = dst + (size_t)(k % NB) * EXPERT_BYTES;
                CK(cudaMemcpy(d, q[k], EXPERT_Q_BYTES, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(d + EXPERT_Q_BYTES, sc[k], EXPERT_BYTES - EXPERT_Q_BYTES, cudaMemcpyHostToDevice));
            }
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("  driver pageable copy map -> VRAM, 1 thread: %.2f GB/s\n", 1024 * EXPERT_BYTES / s / 1e9);
        }
        CK(cudaFree(dst));
        CK(cudaFreeHost(pin));
    }

    if (want(filt, "REG")) {
        printf("REG: can mmapped expert pages be registered for direct DMA (no staging copy)?\n");
        const uint8_t* lo = C.map("model.layers.1.mlp.experts.0.merged_weight");
        const uint8_t* hi = lo;
        for (int e = 0; e < 256; e++) {
            std::string ep = "model.layers.1.mlp.experts." + std::to_string(e) + ".";
            const uint8_t *q = C.map(ep + "merged_weight"), *s = C.map(ep + "qs");
            lo = std::min({lo, q, s}); hi = std::max({hi, q + EXPERT_Q_BYTES, s + (EXPERT_BYTES - EXPERT_Q_BYTES)});
        }
        uintptr_t a = (uintptr_t)lo & ~(uintptr_t)4095, b = ((uintptr_t)hi + 4095) & ~(uintptr_t)4095;
        size_t len = b - a;
        for (size_t o = 0; o < len; o += 4096) { volatile uint8_t x = ((const uint8_t*)a)[o]; (void)x; }  // fault in
        cudaError_t e = cudaHostRegister((void*)a, len, cudaHostRegisterReadOnly);
        printf("  cudaHostRegister(%.0f MB, ReadOnly): %s\n", len / 1e6, cudaGetErrorString(e));
        if (e != cudaSuccess) { cudaGetLastError(); e = cudaHostRegister((void*)a, len, cudaHostRegisterDefault); printf("  default flags: %s\n", cudaGetErrorString(e)); cudaGetLastError(); }
        uint8_t* dst; CK(cudaMalloc(&dst, 64 * EXPERT_Q_BYTES));
        auto timed = [&](const uint8_t* src, size_t stride) {
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 64; i++) CK(cudaMemcpyAsync(dst + i * EXPERT_Q_BYTES, src + i * stride, EXPERT_Q_BYTES, cudaMemcpyHostToDevice));
            CK(cudaDeviceSynchronize());
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        };
        for (int pass = 0; pass < 3; pass++) {
            double ms = timed(lo, EXPERT_Q_BYTES);
            printf("  pass %d: 64 expert copies straight from the map: %.1f ms = %.1f GB/s (%s)\n", pass, ms, 64 * EXPERT_Q_BYTES / ms / 1e6,
                   e == cudaSuccess ? "registered" : "pageable");
        }
        uint8_t* pin; CK(cudaMallocHost(&pin, 64 * EXPERT_Q_BYTES)); memset(pin, 1, 64 * EXPERT_Q_BYTES);
        for (int pass = 0; pass < 2; pass++) {
            double ms = timed(pin, EXPERT_Q_BYTES);
            printf("  reference: same copies from cudaMallocHost memory: %.1f ms = %.1f GB/s\n", ms, 64 * EXPERT_Q_BYTES / ms / 1e6);
        }
        CK(cudaFreeHost(pin));
        if (e == cudaSuccess) cudaHostUnregister((void*)a);
        CK(cudaFree(dst));
    }

    if (want(filt, "SWEEP")) {
        printf("gemv_i8 launch-config sweep on the engine's (fused) shapes; random weights\n");
        struct Sh { const char* n; int O, I; } shapes[] = {
            {"dn qkv+z+a+b 12352x2048", 12352, 2048}, {"attn q+k+v 9216x2048", 9216, 2048}, {"out/o 2048x4096", 2048, 4096},
            {"shared gate+up 1024x2048", 1024, 2048}, {"shared down 2048x512", 2048, 512}, {"lm_head 248320x2048", 248320, 2048}};
        for (auto& sh : shapes) {
            size_t n = (size_t)sh.O * sh.I;
            std::vector<int8_t> hq(n); for (size_t i = 0; i < n; i++) hq[i] = (int8_t)((i * 2654435761u >> 13) % 255 - 127);
            std::vector<float> hs(sh.O, 0.01f), hx(sh.I); for (int i = 0; i < sh.I; i++) hx[i] = sinf(0.1f * i);
            int8_t* q = up(hq); float *s = up(hs), *x = up(hx), *y = dalloc(sh.O), *y0 = dalloc(sh.O);
            gemv_i8<8, 1>(q, s, x, y0, sh.I, sh.O);
            auto ref = down(y0, sh.O);
            double bytes = (double)n + sh.O * 4.0;
            printf(" %s\n", sh.n);
            auto one = [&](const char* nm, auto fn) {
                fn(); auto got = down(y, sh.O);
                double e = 0, mx = 0; for (int i = 0; i < sh.O; i++) { e = std::max(e, (double)fabsf(got[i] - ref[i])); mx = std::max(mx, (double)fabsf(ref[i])); }
                if (e / mx > 1e-5) { printf("  %s WRONG rel %.2e\n", nm, e / mx); g_fail++; }
                bench(nm, bytes, fn, sh.O > 100000 ? 20 : 50);
            };
            one("  W8 R1", [&] { gemv_i8<8, 1>(q, s, x, y, sh.I, sh.O); });
            one("  W8 R2", [&] { gemv_i8<8, 2>(q, s, x, y, sh.I, sh.O); });
            one("  W8 R4", [&] { gemv_i8<8, 4>(q, s, x, y, sh.I, sh.O); });
            one("  W4 R2", [&] { gemv_i8<4, 2>(q, s, x, y, sh.I, sh.O); });
            one("  W4 R1", [&] { gemv_i8<4, 1>(q, s, x, y, sh.I, sh.O); });
            for (void* p : {(void*)q, (void*)s, (void*)x, (void*)y, (void*)y0}) CK(cudaFree(p));
        }
    }

    const int NPOS = 12;  // golden positions p000..p011
    auto pn = [](int p, const char* rest) { char b[64]; snprintf(b, sizeof b, "p%03d_%s", p, rest); return std::string(b); };

    if (want(filt, "K6") || want(filt, "K7")) {
        printf("K6 qk-norm/rope/kv-append + K7 decode attention (L03, chained over %d positions)\n", NPOS);
        const int H = 16, KV = 2, MAXT = 4096, ROT = 64;
        std::string lp = "model.layers.3.";
        half *ln = wup(C, lp + "input_layernorm.weight"), *Wq = wup(C, lp + "self_attn.q_proj.weight"),
             *Wk = wup(C, lp + "self_attn.k_proj.weight"), *Wv = wup(C, lp + "self_attn.v_proj.weight"),
             *Wo = wup(C, lp + "self_attn.o_proj.weight"), *qn = wup(C, lp + "self_attn.q_norm.weight"),
             *kn = wup(C, lp + "self_attn.k_norm.weight");
        float *h = dalloc(D), *q = dalloc(H * 2 * AHD), *k = dalloc(KV * AHD), *v = dalloc(KV * AHD),
              *qo = dalloc(H * AHD), *ag = dalloc(H * AHD), *o = dalloc(D);
        KVCache kvc = kv_alloc(KV, MAXT);
        int* pos_d; CK(cudaMalloc(&pos_d, 4));
        float* part = dalloc(H * 16 * (AHD + 2));
        auto step = [&](int p) {
            set_int_k<<<1, 1>>>(pos_d, p);
            gemv_f16(Wq, h, q, D, H * 2 * AHD); gemv_f16(Wk, h, k, D, KV * AHD); gemv_f16(Wv, h, v, D, KV * AHD);
            attn_decode(q, k, v, qn, kn, qo, kvc, ag, H, KV, pos_d, MAXT, ROT, 1e7f, EPS, 1, 0, 0, part, 16);
            gemv_f16(Wo, ag, o, H * AHD, D);
        };
        double worst_q = 0, worst_o = 0;
        for (int p = 0; p < NPOS; p++) {
            float* x = up(golden(pn(p, "L03_x_in")));
            rmsnorm(x, nullptr, ln, h, D, EPS);
            step(p);
            auto gq = down(qo, H * AHD), rq = golden(pn(p, "L03_attn_q"));
            auto go = down(o, D), ro = golden(pn(p, "L03_attn_out"));
            auto rel = [](const std::vector<float>& g, const std::vector<float>& r) {
                double mx = 0, e = 0; for (size_t i = 0; i < r.size(); i++) { mx = std::max(mx, (double)fabsf(r[i])); e = std::max(e, (double)fabsf(g[i] - r[i])); }
                return e / (mx + 1e-30); };
            worst_q = std::max(worst_q, rel(gq, rq)); worst_o = std::max(worst_o, rel(go, ro));
            // attn_out goes through the int8 KV cache: quantization error, not kernel error (the query is exact)
            if (p == NPOS - 1) { check("L03 p011 query after norm+rope", gq, rq, 1e-5f); check("L03 p011 attn_out (int8 KV)", go, ro, 2e-2f); }
            CK(cudaFree(x));
        }
        printf("  worst rel over all positions: query %.3e  attn_out %.3e (int8 KV)  %s\n", worst_q, worst_o,
               worst_q < 1e-5 && worst_o < 2e-2 ? "PASS" : "FAIL");
        if (!(worst_q < 1e-5 && worst_o < 2e-2)) g_fail++;
        bench("attn layer (proj+core) T=12", (8192.0 + 512 + 512) * D * 2 + 2048.0 * 4096 * 2, [&] { step(NPOS - 1); });
        set_int_k<<<1, 1>>>(pos_d, MAXT - 1);
        for (int ns : {1, 16})
            bench(ns == 1 ? "attn core T=4096, 1 split" : "attn core T=4096, 16 splits", 2.0 * KV * 4096 * AHD, [&] {
                attn_decode_k<<<dim3(H, 1, ns), AHD>>>(qo, q, kvc, ag, H, KV, pos_d, MAXT, 0, part);
                if (ns > 1) attn_combine_k<<<H, AHD>>>(part, q, ag, H, ns, 0); });
    }

    if (want(filt, "K7P")) {
        printf("K7P prefill chunk attention (tensor cores, GQA) vs attn_decode_k per row; random int8 cache\n");
        const int H = 16, KV = 2, MAXT = 17408, LDQ = H * 2 * AHD;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> U(-1.f, 1.f);
        std::vector<int8_t> hk((size_t)KV * MAXT * AHD), hv(hk.size());
        std::vector<float> hks((size_t)KV * MAXT), hvs(hks.size());
        for (auto& x : hk) x = (int8_t)(127 * U(rng)); for (auto& x : hv) x = (int8_t)(127 * U(rng));
        for (auto& x : hks) x = 0.02f * (1.5f + U(rng)); for (auto& x : hvs) x = 0.01f * (1.5f + U(rng));
        KVCache kvc = kv_alloc(KV, MAXT);
        CK(cudaMemcpy(kvc.k, hk.data(), hk.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(kvc.v, hv.data(), hv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(kvc.ks, hks.data(), hks.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(kvc.vs, hvs.data(), hvs.size() * 4, cudaMemcpyHostToDevice));
        const int TM = 1024;
        std::vector<float> hq((size_t)TM * H * AHD), hg((size_t)TM * LDQ);
        for (auto& x : hq) x = 3.f * U(rng); for (auto& x : hg) x = U(rng);
        float *qo = up(hq), *qg = up(hg), *o1 = dalloc((size_t)TM * H * AHD), *o2 = dalloc((size_t)TM * H * AHD);
        int* pos_d; CK(cudaMalloc(&pos_d, 4));
        for (auto [T, p0] : {std::pair{37, 0}, std::pair{37, 1000}, std::pair{1024, 3072}}) {
            set_int_k<<<1, 1>>>(pos_d, p0);
            attn_decode_k<<<dim3(H, T, 1), AHD>>>(qo, qg, kvc, o1, H, KV, pos_d, MAXT, LDQ, nullptr);
            attn_prefill(qo, qg, kvc, o2, H, KV, pos_d, MAXT, LDQ, T, 0);
            char lbl[64]; snprintf(lbl, sizeof lbl, "T=%d at pos %d vs per-row", T, p0);
            check(lbl, down(o2, (size_t)T * H * AHD), down(o1, (size_t)T * H * AHD), 5e-3f);  // fp16 Q and P
        }
        for (int p0 : {3072, 15360}) {
            set_int_k<<<1, 1>>>(pos_d, p0);
            char lbl[64]; snprintf(lbl, sizeof lbl, "1024 rows at pos %d: per-row", p0);
            bench(lbl, 2.0 * KV * (p0 + 512) * AHD, [&] { attn_decode_k<<<dim3(H, TM, 1), AHD>>>(qo, qg, kvc, o1, H, KV, pos_d, MAXT, LDQ, nullptr); }, 3);
            snprintf(lbl, sizeof lbl, "1024 rows at pos %d: prefill", p0);
            bench(lbl, 2.0 * KV * (p0 + 512) * AHD, [&] { attn_prefill(qo, qg, kvc, o2, H, KV, pos_d, MAXT, LDQ, TM, 0); }, 3);
        }
    }

    if (want(filt, "K8")) {
        printf("K8 Gated DeltaNet (L00, chained over %d positions)\n", NPOS);
        const int C8 = 8192, VH = 32, KH = 16;
        std::string lp = "model.layers.0.", la = lp + "linear_attn.";
        half *ln = wup(C, lp + "input_layernorm.weight"), *Wqkv = wup(C, la + "in_proj_qkv.weight"),
             *Wz = wup(C, la + "in_proj_z.weight"), *Wa = wup(C, la + "in_proj_a.weight"), *Wb = wup(C, la + "in_proj_b.weight"),
             *Wo = wup(C, la + "out_proj.weight"), *cw = wup(C, la + "conv1d.weight"), *Al = wup(C, la + "A_log"),
             *dtb = wup(C, la + "dt_bias"), *nw = wup(C, la + "norm.weight");
        float *h = dalloc(D), *qkv = dalloc(C8), *z = dalloc(VH * DNV), *a = dalloc(VH), *b = dalloc(VH),
              *conv = dalloc(C8), *o = dalloc(VH * DNV), *out = dalloc(D);
        float *ring = dalloc(C8 * 3), *S = dalloc((size_t)VH * DNK * DNV);
        CK(cudaMemset(ring, 0, C8 * 3 * 4)); CK(cudaMemset(S, 0, (size_t)VH * DNK * DNV * 4));
        auto step = [&] {
            gemv_f16(Wqkv, h, qkv, D, C8); gemv_f16(Wz, h, z, D, VH * DNV);
            gemv_f16(Wa, h, a, D, VH); gemv_f16(Wb, h, b, D, VH);
            deltanet_core(qkv, z, a, b, 0, cw, Al, dtb, nw, ring, S, conv, o, C8, 4, VH, KH, EPS);
            gemv_f16(Wo, o, out, VH * DNV, D);
        };
        double wc = 0, wo = 0, wout = 0;
        auto rel = [](const std::vector<float>& g, const std::vector<float>& r) {
            double mx = 0, e = 0; for (size_t i = 0; i < r.size(); i++) { mx = std::max(mx, (double)fabsf(r[i])); e = std::max(e, (double)fabsf(g[i] - r[i])); }
            return e / (mx + 1e-30); };
        for (int p = 0; p < NPOS; p++) {
            float* x = up(golden(pn(p, "L00_x_in")));
            rmsnorm(x, nullptr, ln, h, D, EPS);
            step();
            wc = std::max(wc, rel(down(conv, C8), golden(pn(p, "L00_dn_conv"))));
            wo = std::max(wo, rel(down(o, VH * DNV), golden(pn(p, "L00_dn_o"))));
            auto go = down(out, D), ro = golden(pn(p, "L00_dn_out"));
            wout = std::max(wout, rel(go, ro));
            if (p == NPOS - 1) check("L00 p011 dn_out", go, ro, 1e-4f);
            CK(cudaFree(x));
        }
        bool ok = wc < 1e-5 && wo < 1e-4 && wout < 1e-4;
        printf("  worst rel over all positions: conv %.3e  gated-norm o %.3e  dn_out %.3e  %s\n", wc, wo, wout, ok ? "PASS" : "FAIL");
        if (!ok) g_fail++;
        bench("deltanet layer (proj+core)", (8192.0 + 4096 + 64) * D * 2 + 2048.0 * 4096 * 2 + 2.0 * VH * DNK * DNV * 4, step);
        bench("deltanet core (conv+recur)", 2.0 * VH * DNK * DNV * 4, [&] {
            deltanet_core(qkv, z, a, b, 0, cw, Al, dtb, nw, ring, S, conv, o, C8, 4, VH, KH, EPS); });
    }

    if (want(filt, "K9")) {
        printf("K9 argmax (all %d positions)\n", NPOS);
        const int V = 248320;
        float* sv = dalloc(ARGMAX_BLOCKS + 1); int *si, *res; CK(cudaMalloc(&si, ARGMAX_BLOCKS * 4)); CK(cudaMalloc(&res, 4));
        int bad = 0; float* lg = nullptr;
        for (int p = 0; p < NPOS; p++) {
            auto ref = golden(pn(p, "logits"));
            if (lg) CK(cudaFree(lg));
            lg = up(ref);
            argmax(lg, V, sv, si, res);
            int g; CK(cudaMemcpy(&g, res, 4, cudaMemcpyDeviceToHost));
            if (g != int(std::max_element(ref.begin(), ref.end()) - ref.begin())) bad++;
        }
        printf("  argmax matches host on %d/%d positions %s\n", NPOS - bad, NPOS, bad ? "FAIL" : "PASS");
        if (bad) g_fail++;
        bench("argmax 248320", V * 4.0, [&] { argmax(lg, V, sv, si, res); });
    }

    printf(g_fail ? "\n%d FAILED\n" : "\nALL PASS\n", g_fail);
    return g_fail != 0;
}
