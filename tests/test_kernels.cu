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

    printf(g_fail ? "\n%d FAILED\n" : "\nALL PASS\n", g_fail);
    return g_fail != 0;
}
