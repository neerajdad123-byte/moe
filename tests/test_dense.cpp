// CPU test of src/dense_ovr.h against tools/quant_dense.py: `python tools/quant_dense.py --selftest` writes
// build/dense_test (int4 / int6, gs 64 / 32, symmetric / asymmetric, with and without AWQ scales) and the values
// Python dequantizes them to; the engine's decoder must give exactly those. build.bat -> build\test_dense.exe
#include <cmath>
#include <cstdio>
#include <vector>
#include "../src/dense_ovr.h"

int main() {
    const int O = 96, I = 512, NT = 4;
    FILE* f = fopen("build/dense_test/ref.f32", "rb");
    if (!f) { printf("no build/dense_test (run python tools/quant_dense.py --selftest first)\n"); return 1; }
    std::vector<float> ref((size_t)NT * O * I);
    if (fread(ref.data(), 4, ref.size(), f) != ref.size()) { printf("short ref.f32\n"); return 1; }
    fclose(f);
    Container C("build/dense_test");
    int fails = 0;
    const char* what[NT] = {"int4 gs64 sym (rtn)", "int6 gs64 sym (gptq)", "int4 gs32 asym + AWQ scales", "int6 gs32 asym + AWQ (awq+gptq)"};
    for (int t = 0; t < NT; t++) {
        long long bad = 0; double worst = 0;
        dequant_packed(C, "t" + std::to_string(t), O, I, [&](int64_t k, float w) {
            const float r = ref[(size_t)t * O * I + k];
            worst = std::max(worst, (double)std::fabs(w - r));
            bad += w != r;
        });
        printf("%s %-34s mismatches %lld, max abs diff %.3g\n", bad ? "FAIL" : "ok  ", what[t], bad, worst);
        fails += bad != 0;
    }
    printf(fails ? "test_dense: %d FAILED\n" : "test_dense: ALL PASS\n", fails);
    return fails != 0;
}
