// CPU unit test of src/gguf.h: dequantization against llama.cpp's reference (tests/gguf_blocks.bin from
// tools/make_gguf_fixture.py: real Q4_K/Q6_K/Q8_0/F32 blocks, random F16/BF16/Q5_K), header parsing of a small
// synthetic GGUF (and refusal of a truncated one), the DeltaNet value-head un-permutation and the int4 requantizer.
// build.bat -> build\test_gguf.exe; run from the repo root.
#include <random>
#include "../src/gguf.h"

static int fails = 0;
static void check(bool ok, const std::string& what) { printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str()); fails += !ok; }

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);  // every line out before a crash
    {  // dequantization, every supported type
        FILE* f = fopen("tests/gguf_blocks.bin", "rb");
        check(f != nullptr, "fixture tests/gguf_blocks.bin opens");
        uint32_t h[2];
        while (f && fread(h, 4, 2, f) == 2) {
            int bn, bb;
            check(gg_block(h[0], bn, bb), "type " + gg_type_name(h[0]) + " is supported");
            std::vector<uint8_t> raw((size_t)h[1] / bn * bb);
            std::vector<float> want(h[1]), got(h[1]);
            if (fread(raw.data(), 1, raw.size(), f) != raw.size() || fread(want.data(), 4, h[1], f) != h[1]) { check(false, "fixture complete"); break; }
            gg_dequant(h[0], raw.data(), h[1], got.data());
            float amax = 0, err = 0;
            for (uint32_t i = 0; i < h[1]; i++) { amax = std::max(amax, std::fabs(want[i])); err = std::max(err, std::fabs(got[i] - want[i])); }
            check(amax > 0 && err <= 1e-6f * amax, "dequant " + gg_type_name(h[0]) + " matches gguf-py (" + std::to_string(h[1]) + " values, max err " + std::to_string(err / amax) + " of max)");
        }
        if (f) fclose(f);
    }
    {  // half <-> float: every finite half round-trips; ties round to even
        bool ok = true;
        for (uint32_t h = 0; h < 65536; h++)
            if ((h & 0x7C00) != 0x7C00) ok &= gg_f2h(gg_half((uint16_t)h)) == h;
        ok &= gg_f2h(1.0f + 1.0f / 2048) == 0x3C00 && gg_f2h(1.0f + 3.0f / 2048) == 0x3C02 && gg_f2h(70000.0f) == 0x7C00;
        check(ok, "half <-> float: exact round trip, round to nearest even, overflow to inf");
    }
    const std::string path = "build/test_gguf_tiny.gguf";
    {  // a small GGUF v3: kv of each kind used, an F32 and a Q8_0 tensor
        std::string b = "GGUF";
        auto u32 = [&](uint32_t v) { b.append((const char*)&v, 4); };
        auto u64 = [&](uint64_t v) { b.append((const char*)&v, 8); };
        auto str = [&](const std::string& s) { u64(s.size()); b += s; };
        u32(3); u64(2); u64(4);
        str("general.architecture"); u32(8); str("qwen35moe");
        str("qwen35moe.block_count"); u32(4); u32(40);
        str("tokenizer.ggml.tokens"); u32(9); u32(8); u64(2); str("a"); str("\xc4\xa0" "b");
        str("qwen35moe.rope.dimension_sections"); u32(9); u32(5); u64(4); u32(11); u32(11); u32(10); u32(0);
        str("w"); u32(2); u64(4); u64(2); u32(GG_F32); u64(0);
        str("q"); u32(1); u64(32); u32(GG_Q8_0); u64(32);
        b.append((32 - b.size() % 32) % 32, '\0');
        for (int i = 0; i < 8; i++) { float v = i * 0.5f; b.append((const char*)&v, 4); }
        const uint16_t d = gg_f2h(0.25f);
        b.append((const char*)&d, 2);
        for (int i = 0; i < 32; i++) b += (char)(int8_t)(i - 16);
        FILE* f = fopen(path.c_str(), "wb"); fwrite(b.data(), 1, b.size(), f); fclose(f);
        Gguf g(path);
        check(g.get("general.architecture")->str == "qwen35moe" && g.get("qwen35moe.block_count")->num == 40, "header: string and u32 values");
        check(g.get("tokenizer.ggml.tokens")->strs.size() == 2 && g.get("tokenizer.ggml.tokens")->strs[1] == "\xc4\xa0" "b", "header: string array");
        check(g.get("qwen35moe.rope.dimension_sections")->nums == std::vector<double>{11, 11, 10, 0}, "header: i32 array");
        std::vector<float> w = g.all("w"), q = g.all("q");
        check(g.at("w").ne == std::vector<int64_t>{4, 2} && w[7] == 3.5f, "tensor: F32 values at the aligned data offset");
        check(q[0] == -4.0f && q[31] == 3.75f, "tensor: Q8_0 values");
        const std::string cut = path + ".cut", cut2 = path + ".cut2";  // g keeps its file mapped (for the process lifetime)
        f = fopen(cut.c_str(), "wb"); fwrite(b.data(), 1, b.size() - 10, f); fclose(f);
        bool threw = false;
        try { Gguf t(cut); } catch (const std::exception&) { threw = true; }
        check(threw, "truncated file (tensor past the end) is refused");
        f = fopen(cut2.c_str(), "wb"); fwrite("GGUF\x03\0\0\0\xff\xff", 1, 10, f); fclose(f);
        threw = false;
        try { Gguf t(cut2); } catch (const std::exception&) { threw = true; }
        check(threw, "truncated header is refused");
        std::string z = b;  // tensor "w" with a zero dimension: refused, not a division by zero
        z.replace(z.find(std::string("w\x02\0\0\0", 5)) + 5, 8, std::string(8, '\0'));
        f = fopen((cut + "3").c_str(), "wb"); fwrite(z.data(), 1, z.size(), f); fclose(f);
        threw = false;
        try { Gguf t(cut + "3"); } catch (const std::exception&) { threw = true; }
        check(threw, "zero tensor dimension is refused");
    }
    {  // llama.cpp puts HF value head 2 * (j % 16) + j / 16 at GGUF head j; un-permuting must invert that exactly
        bool inv = true;
        for (int j = 0; j < 32; j++) inv &= gg_vhead_src(2 * (j % 16) + j / 16) == j;
        std::vector<float> w(2 + 32 * 3 * 2);  // 2 leading rows (q/k part), then 32 heads x 3 rows x 2 cols
        for (size_t i = 0; i < w.size(); i++) w[i] = (float)i;
        std::vector<float> u(w);
        gg_unperm_rows(u, 1, 3, 2);
        bool ok = u[0] == 0 && u[1] == 1;
        for (int h = 0; h < 32; h++)
            for (int r = 0; r < 6; r++) { const int j = gg_vhead_src(h); ok &= u[2 + h * 6 + r] == w[2 + j * 6 + r]; }
        check(inv && ok, "value-head un-permutation inverts llama.cpp's reorder and leaves the leading rows");
    }
    {  // zero-point experts: Q4_K is carried over exactly (the same bits); other types refit per 32
        FILE* f = fopen("tests/gguf_blocks.bin", "rb");
        uint32_t h[2];
        while (f && fread(h, 4, 2, f) == 2) {
            int bn, bb;
            gg_block(h[0], bn, bb);
            std::vector<uint8_t> raw((size_t)h[1] / bn * bb);
            std::vector<float> want(h[1]);
            fread(raw.data(), 1, raw.size(), f); fread(want.data(), 4, h[1], f);
            if (h[0] != GG_Q4_K && h[0] != GG_Q6_K) continue;
            std::vector<uint8_t> qb(h[1] / 2); std::vector<uint8_t> cs(h[1] / 32 * 2); std::vector<uint16_t> sup(h[1] / 256 * 2); std::vector<float> tmp(h[1]);
            gg_zp_row(h[0], raw.data(), (int)h[1], qb.data(), cs.data(), sup.data(), tmp.data());
            double amax = 0, err = 0, ss = 0, ee = 0;
            for (uint32_t i = 0; i < h[1]; i++) {
                const float v = gg_zp_at(qb.data(), cs.data(), sup.data(), (int)i);
                amax = std::max(amax, (double)std::fabs(want[i])); err = std::max(err, (double)std::fabs(v - want[i]));
                ss += (double)want[i] * want[i]; ee += (double)(v - want[i]) * (v - want[i]);
            }
            if (h[0] == GG_Q4_K) check(err <= 1e-6 * amax, "zero point: Q4_K experts carried over exactly (max err " + std::to_string(err / amax) + " of max)");
            else check(std::sqrt(ee / ss) < 0.1, "zero point: Q6_K requantized to 4 bits, relative RMS " + std::to_string(std::sqrt(ee / ss)));
        }
        if (f) fclose(f);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd;
        bool never_worse = true, in_range = true;
        double e_ls = 0, e_mm = 0;
        for (int g = 0; g < 4000; g++) {
            float x[32];
            for (int i = 0; i < 32; i++) x[i] = g % 2 ? nd(rng) : 0.01f * (int)(rng() % 64) - 0.3f;  // odd: normal, even: a Q6_K-like grid
            uint8_t u[32]; float s, m;
            gg_zp_group(x, u, s, m);
            float lo = x[0], hi = x[0];
            for (float v : x) { lo = std::min(lo, v); hi = std::max(hi, v); }
            const float s0 = (hi - lo) / 15;
            double el = 0, em = 0;
            for (int i = 0; i < 32; i++) {
                in_range &= u[i] <= 15;
                el += (x[i] - s * u[i] - m) * (x[i] - s * u[i] - m);
                const int q0 = std::clamp((int)std::nearbyint((x[i] - lo) / s0), 0, 15);
                em += (x[i] - s0 * q0 - lo) * (x[i] - s0 * q0 - lo);
            }
            never_worse &= el <= em * (1 + 1e-5);
            e_ls += el; e_mm += em;
        }
        check(in_range && never_worse, "zero point: least-squares refit never worse than min/max rounding (total sq. error " +
                                       std::to_string(e_ls / e_mm) + "x)");
    }
    printf(fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
