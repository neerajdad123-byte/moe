// Kernel correctness (vs ref/golden from qwen36_ref.py) + bandwidth bench.
//   build\test_kernels.exe [filter]
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "../src/st.h"
#include "../src/kernels.cuh"

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
static float* dalloc(size_t n) { float* d; CK(cudaMalloc(&d, n * 4)); return d; }
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
            router_topk(logits, E, K, false, idx, w);
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
            experts_forward(slots, w, K, h, act, out);
            shared_expert_add(Wg, Wu, Wd, Ws, h, out, scr, D, SI);
            check((std::string(pf) + "routed+shared -> moe_out").c_str(), down(out, D), golden(pf + std::string("moe_out")), 1e-5f);
            if (L == 0) {
                bench("router gemv 256x2048 + topk", 256.0 * D * 2, [&] { gemv_f16(Wr, h, logits, D, E); router_topk(logits, E, K, false, idx, w); });
                bench("experts x8 int4 gs64", (double)K * EXPERT_BYTES, [&] { experts_forward(slots, w, K, h, act, out); });
                bench("shared expert F16", 3.0 * SI * D * 2, [&] { shared_expert_add(Wg, Wu, Wd, Ws, h, out, scr, D, SI); });
            }
            for (void* p : {(void*)Wr, (void*)h, (void*)Wg, (void*)Wu, (void*)Wd, (void*)Ws}) CK(cudaFree(p));
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
        float *Kc = dalloc((size_t)KV * MAXT * AHD), *Vc = dalloc((size_t)KV * MAXT * AHD);
        auto step = [&](int p) {
            gemv_f16(Wq, h, q, D, H * 2 * AHD); gemv_f16(Wk, h, k, D, KV * AHD); gemv_f16(Wv, h, v, D, KV * AHD);
            attn_decode(q, k, v, qn, kn, qo, Kc, Vc, ag, H, KV, p, MAXT, ROT, 1e7f, EPS);
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
            if (p == NPOS - 1) { check("L03 p011 query after norm+rope", gq, rq, 1e-5f); check("L03 p011 attn_out", go, ro, 1e-5f); }
            CK(cudaFree(x));
        }
        printf("  worst rel over all positions: query %.3e  attn_out %.3e  %s\n", worst_q, worst_o,
               worst_q < 1e-5 && worst_o < 1e-5 ? "PASS" : "FAIL");
        if (!(worst_q < 1e-5 && worst_o < 1e-5)) g_fail++;
        bench("attn layer (proj+core) T=12", (8192.0 + 512 + 512) * D * 2 + 2048.0 * 4096 * 2, [&] { step(NPOS - 1); });
        CK(cudaFuncSetAttribute(attn_decode_k, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXT * 4));
        bench("attn core T=4096", 2.0 * KV * 4096 * AHD * 4, [&] {
            attn_decode_k<<<H, AHD, MAXT * 4>>>(qo, q, Kc, Vc, ag, H, KV, MAXT, MAXT); });
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
            deltanet_core(qkv, z, a, b, cw, Al, dtb, nw, ring, S, conv, o, C8, 4, VH, KH, EPS);
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
            deltanet_core(qkv, z, a, b, cw, Al, dtb, nw, ring, S, conv, o, C8, 4, VH, KH, EPS); });
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
