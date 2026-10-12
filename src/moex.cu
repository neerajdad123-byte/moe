// MoEx v2 decode engine (Phase 2: everything on GPU, experts fetched into VRAM slots on miss).
//   moex.exe --ids 760,6511,314 [--n 32] [--ctx 2048] [--stop 248044,248046] [--profile]
//   moex.exe --check ref/golden          logits vs the PyTorch reference at every golden position
//   --f16                                 keep dense weights F16 (exact); default is int8 per-row
// Generated token ids go to stdout one per line; stats go to stderr. Tokenizing is moex.py's job.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "st.h"
#include "kernels.cuh"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Qwen3.6-35B-A3B geometry (checked against the container shapes at load).
constexpr int D = 2048, NL = 40, E = 256, TOPK = 8, SI = 512, V = 248320;
constexpr int AH = 16, AKV = 2, ROT = 64;            // attention
constexpr int DC = 8192, DVH = 32, DKH = 16, CONVK = 4;  // DeltaNet
constexpr float EPS = 1e-6f, THETA = 1e7f;
static bool is_attn(int l) { return l % 4 == 3; }

struct LayerW {
    half *in_ln, *post_ln, *router, *sh_gate, *qn, *kn, *a, *b, *conv, *nw, *alog, *dtb;  // small: F16 always
    Mat sh_g, sh_u, sh_d, q, k, v, o, qkv, z, out;                                        // big: int8 unless --f16
    float *K, *V, *ring, *S;                                          // state
};

struct Engine {
    Container C;
    std::vector<LayerW> L;
    Mat lm_head;
    half* norm;
    bool f16 = false;
    half* qtmp = nullptr;  // F16 staging for quantization, freed after load
    const half* embed;  // mmapped, stays in RAM
    int ctx;
    // scratch
    float *x, *h, *tmp, *moe, *big, *q, *k, *v, *qo, *ag, *z, *a, *b, *conv, *o, *act, *scr, *logits, *rw, *amv;
    int *ridx, *ams, *tok_d;
    const uint8_t** slots_d;
    // expert slot cache (LRU)
    uint8_t* arena = nullptr;
    int nslots = 0;
    std::vector<int> slot_of, owner;
    std::vector<uint64_t> last;
    uint64_t clock = 0;
    long long hits = 0, misses = 0;
    double load_s = 0;
    // profile (synchronizing, so it perturbs the speed it measures)
    bool prof = false;
    double tp[8] = {};

    explicit Engine(const char* root, int ctx_) : C(root), ctx(ctx_) {}

    half* wup(const std::string& n, int64_t numel) {
        const StTensor& s = C.at(n);
        if ((int64_t)s.bytes != numel * 2) { fprintf(stderr, "shape mismatch %s\n", n.c_str()); exit(1); }
        half* d; CK(cudaMalloc(&d, s.bytes));
        CK(cudaMemcpy(d, C.map(n), s.bytes, cudaMemcpyHostToDevice));
        return d;
    }
    Mat wmat(const std::string& n, int64_t O, int64_t I) {
        if (f16) return {wup(n, O * I)};
        const StTensor& st = C.at(n);
        if ((int64_t)st.bytes != O * I * 2) { fprintf(stderr, "shape mismatch %s\n", n.c_str()); exit(1); }
        CK(cudaMemcpy(qtmp, C.map(n), st.bytes, cudaMemcpyHostToDevice));
        int8_t* q; float* sc;
        CK(cudaMalloc(&q, O * I)); CK(cudaMalloc(&sc, O * 4));
        quantize_rows_i8_k<<<(int)O, 256>>>(qtmp, q, sc, (int)I);
        CK(cudaGetLastError());
        return {nullptr, q, sc};
    }
    static float* falloc(size_t n, bool zero = false) {
        float* d; CK(cudaMalloc(&d, n * 4)); if (zero) CK(cudaMemset(d, 0, n * 4)); return d;
    }

    void load() {
        double t0 = now();
        embed = (const half*)C.map("model.embed_tokens.weight");
        if (!f16) CK(cudaMalloc(&qtmp, (size_t)V * D * 2));  // largest matrix
        lm_head = wmat("lm_head.weight", V, D);
        norm = wup("model.norm.weight", D);
        L.resize(NL);
        for (int i = 0; i < NL; i++) {
            std::string p = "model.layers." + std::to_string(i) + ".", m = p + "mlp.";
            LayerW& w = L[i];
            w.in_ln = wup(p + "input_layernorm.weight", D);
            w.post_ln = wup(p + "post_attention_layernorm.weight", D);
            w.router = wup(m + "gate.weight", (int64_t)E * D);
            w.sh_g = wmat(m + "shared_expert.gate_proj.weight", SI, D);
            w.sh_u = wmat(m + "shared_expert.up_proj.weight", SI, D);
            w.sh_d = wmat(m + "shared_expert.down_proj.weight", D, SI);
            w.sh_gate = wup(m + "shared_expert_gate.weight", D);
            if (is_attn(i)) {
                std::string a = p + "self_attn.";
                w.q = wmat(a + "q_proj.weight", AH * 2 * AHD, D);
                w.k = wmat(a + "k_proj.weight", AKV * AHD, D);
                w.v = wmat(a + "v_proj.weight", AKV * AHD, D);
                w.o = wmat(a + "o_proj.weight", D, AH * AHD);
                w.qn = wup(a + "q_norm.weight", AHD);
                w.kn = wup(a + "k_norm.weight", AHD);
                w.K = falloc((size_t)AKV * ctx * AHD);
                w.V = falloc((size_t)AKV * ctx * AHD);
            } else {
                std::string a = p + "linear_attn.";
                w.qkv = wmat(a + "in_proj_qkv.weight", DC, D);
                w.z = wmat(a + "in_proj_z.weight", DVH * DNV, D);
                w.a = wup(a + "in_proj_a.weight", (int64_t)DVH * D);
                w.b = wup(a + "in_proj_b.weight", (int64_t)DVH * D);
                w.out = wmat(a + "out_proj.weight", D, DVH * DNV);
                w.conv = wup(a + "conv1d.weight", (int64_t)DC * CONVK);
                w.nw = wup(a + "norm.weight", DNV);
                w.alog = wup(a + "A_log", DVH);
                w.dtb = wup(a + "dt_bias", DVH);
                w.ring = falloc((size_t)DC * (CONVK - 1), true);
                w.S = falloc((size_t)DVH * DNK * DNV, true);
            }
        }
        if (qtmp) { CK(cudaDeviceSynchronize()); CK(cudaFree(qtmp)); qtmp = nullptr; }
        x = falloc(D); h = falloc(D); tmp = falloc(D); moe = falloc(D);
        q = falloc(AH * 2 * AHD); k = falloc(AKV * AHD); v = falloc(AKV * AHD); qo = falloc(AH * AHD); ag = falloc(AH * AHD);
        big = falloc(DC); z = falloc(DVH * DNV); a = falloc(DVH); b = falloc(DVH); conv = falloc(DC); o = falloc(DVH * DNV);
        act = falloc(TOPK * EI); scr = falloc(2 * SI + D + 1); logits = falloc(V); rw = falloc(TOPK);
        amv = falloc(ARGMAX_BLOCKS + 1);
        CK(cudaMalloc(&ridx, TOPK * 4)); CK(cudaMalloc(&ams, ARGMAX_BLOCKS * 4)); CK(cudaMalloc(&tok_d, 4));
        CK(cudaMalloc(&slots_d, TOPK * sizeof(void*)));
        if (ctx * 4 > 48 * 1024)
            CK(cudaFuncSetAttribute(attn_decode_k, cudaFuncAttributeMaxDynamicSharedMemorySize, ctx * 4));
        // every byte of VRAM left (minus headroom for the driver) becomes expert slots
        size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = 320ull << 20;
        nslots = (int)((fr - std::min(fr, reserve)) / EXPERT_BYTES);
        if (nslots < 2 * TOPK) { fprintf(stderr, "only %d expert slots fit; need >= %d\n", nslots, 2 * TOPK); exit(1); }
        CK(cudaMalloc(&arena, (size_t)nslots * EXPERT_BYTES));
        slot_of.assign(NL * E, -1); owner.assign(nslots, -1); last.assign(nslots, 0);
        fprintf(stderr, "[load] %.1f s, dense %s, %d expert slots (%.2f GB, %.1f%% of %d experts), ctx %d\n",
                now() - t0, f16 ? "F16" : "int8", nslots, nslots * (double)EXPERT_BYTES / 1e9, 100.0 * nslots / (NL * E), NL * E, ctx);
    }

    // Make the layer's selected experts resident and return their slot pointers.
    // LRU: this layer's experts are stamped first, so a miss never evicts a sibling.
    void ensure(int l, const int* ids, const uint8_t** out) {
        clock++;
        for (int j = 0; j < TOPK; j++) { int s = slot_of[l * E + ids[j]]; if (s >= 0) last[s] = clock; }
        for (int j = 0; j < TOPK; j++) {
            int key = l * E + ids[j], s = slot_of[key];
            if (s >= 0) { hits++; out[j] = arena + (size_t)s * EXPERT_BYTES; continue; }
            misses++;
            s = int(std::min_element(last.begin(), last.end()) - last.begin());
            if (owner[s] >= 0) slot_of[owner[s]] = -1;
            owner[s] = key; slot_of[key] = s; last[s] = clock;
            std::string ep = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(ids[j]) + ".";
            uint8_t* dst = arena + (size_t)s * EXPERT_BYTES;
            double t0 = now();
            // shortcut: synchronous pageable copy from the mmap; Phase 3 replaces misses with CPU compute + async prefetch
            CK(cudaMemcpy(dst, C.map(ep + "merged_weight"), EXPERT_Q_BYTES, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dst + EXPERT_Q_BYTES, C.map(ep + "qs"), EXPERT_BYTES - EXPERT_Q_BYTES, cudaMemcpyHostToDevice));
            load_s += now() - t0;
            out[j] = dst;
        }
    }

    void mark(int i, double& t) { if (prof) { CK(cudaDeviceSynchronize()); double n = now(); tp[i] += n - t; t = n; } }

    // One token at position `pos`. If want_logits, logits[] is filled and the greedy id returned.
    int step(int tok, int pos, bool want_logits) {
        double t = now();
        std::vector<float> e(D);
        for (int i = 0; i < D; i++) e[i] = __half2float(embed[(size_t)tok * D + i]);
        CK(cudaMemcpy(x, e.data(), D * 4, cudaMemcpyHostToDevice));
        const float* pending = nullptr;  // previous MoE output, added to x by the next norm
        for (int l = 0; l < NL; l++) {
            LayerW& w = L[l];
            rmsnorm(x, pending, w.in_ln, h, D, EPS);
            if (is_attn(l)) {
                gemv(w.q, h, q, D, AH * 2 * AHD); gemv(w.k, h, k, D, AKV * AHD); gemv(w.v, h, v, D, AKV * AHD);
                attn_decode(q, k, v, w.qn, w.kn, qo, w.K, w.V, ag, AH, AKV, pos, ctx, ROT, THETA, EPS);
                gemv(w.o, ag, tmp, AH * AHD, D);
                mark(0, t);
            } else {
                gemv(w.qkv, h, big, D, DC); gemv(w.z, h, z, D, DVH * DNV);
                gemv_f16(w.a, h, a, D, DVH); gemv_f16(w.b, h, b, D, DVH);
                deltanet_core(big, z, a, b, w.conv, w.alog, w.dtb, w.nw, w.ring, w.S, conv, o, DC, CONVK, DVH, DKH, EPS);
                gemv(w.out, o, tmp, DVH * DNV, D);
                mark(1, t);
            }
            rmsnorm(x, tmp, w.post_ln, h, D, EPS);
            gemv_f16(w.router, h, scr, D, E);
            router_topk(scr, E, TOPK, false, ridx, rw);
            int ids[TOPK];
            CK(cudaMemcpy(ids, ridx, sizeof ids, cudaMemcpyDeviceToHost));  // per-layer sync: the cache needs the ids
            mark(2, t);
            const uint8_t* sp[TOPK];
            ensure(l, ids, sp);
            CK(cudaMemcpy(slots_d, sp, sizeof sp, cudaMemcpyHostToDevice));
            mark(3, t);
            experts_forward(slots_d, rw, TOPK, h, act, moe);
            mark(4, t);
            shared_expert_add(w.sh_g, w.sh_u, w.sh_d, w.sh_gate, h, moe, scr, D, SI);
            mark(5, t);
            pending = moe;
        }
        // the last MoE output is folded into x here, or by the next token's first norm
        rmsnorm(x, pending, norm, h, D, EPS);
        if (!want_logits) return -1;
        gemv(lm_head, h, logits, D, V);
        argmax(logits, V, amv, ams, tok_d);
        int id; CK(cudaMemcpy(&id, tok_d, 4, cudaMemcpyDeviceToHost));
        mark(6, t);
        return id;
    }
};

static std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    for (size_t p = 0; p < s.size();) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) v.push_back(std::stoi(s.substr(p, q - p))); p = q + 1; }
    return v;
}

static std::vector<float> read_f32(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return {};
    _fseeki64(f, 0, SEEK_END); size_t n = _ftelli64(f) / 4; _fseeki64(f, 0, SEEK_SET);
    std::vector<float> v(n); fread(v.data(), 4, n, f); fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const char* model = getenv("QWEN36_MODEL") ? getenv("QWEN36_MODEL") : "C:\\models\\qwen36_i4_gs64";
    std::vector<int> ids, stop;
    int n = 32, ctx = 2048;
    std::string check;
    bool prof = false, f16 = false;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", s.c_str()); exit(2); } return std::string(argv[++i]); };
        if (s == "--ids") ids = parse_ids(next());
        else if (s == "--n") n = std::stoi(next());
        else if (s == "--ctx") ctx = std::stoi(next());
        else if (s == "--stop") stop = parse_ids(next());
        else if (s == "--check") check = next();
        else if (s == "--profile") prof = true;
        else if (s == "--f16") f16 = true;
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (!check.empty()) {  // ids come from the golden run's meta.json
        FILE* f = fopen((check + "/meta.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s/meta.json\n", check.c_str()); return 1; }
        std::string m(1 << 16, '\0'); m.resize(fread(m.data(), 1, m.size(), f)); fclose(f);
        size_t a = m.find('[', m.find("\"ids\"")) + 1, b = m.find(']', a);
        ids = parse_ids(m.substr(a, b - a));
    }
    if (ids.empty()) { fprintf(stderr, "usage: moex --ids 1,2,3 [--n N] | --check ref/golden\n"); return 2; }

    Engine eng(model, ctx);
    eng.f16 = f16;
    eng.load();

    if (!check.empty()) {
        int pos = 0, tok = ids[0], bad = 0; double worst = 0, worst_kl = 0;
        for (;; pos++) {
            auto ref = read_f32(check + "/p" + std::string(pos < 10 ? "00" : pos < 100 ? "0" : "") + std::to_string(pos) + "_logits.bin");
            if (ref.empty()) break;
            int id = eng.step(tok, pos, true);
            std::vector<float> got(V); CK(cudaMemcpy(got.data(), eng.logits, V * 4, cudaMemcpyDeviceToHost));
            double mx = 0, err = 0;
            for (int i = 0; i < V; i++) { mx = std::max(mx, (double)fabsf(ref[i])); err = std::max(err, (double)fabsf(got[i] - ref[i])); }
            int rid = int(std::max_element(ref.begin(), ref.end()) - ref.begin());
            // KL(ref || got) of the next-token distributions
            double mr = ref[rid], mg = *std::max_element(got.begin(), got.end()), zr = 0, zg = 0, kl = 0;
            for (int i = 0; i < V; i++) { zr += exp(ref[i] - mr); zg += exp(got[i] - mg); }
            for (int i = 0; i < V; i++) { double lp = ref[i] - mr - log(zr), lq = got[i] - mg - log(zg); kl += exp(lp) * (lp - lq); }
            worst = std::max(worst, err / mx); worst_kl = std::max(worst_kl, kl); bad += id != rid;
            fprintf(stderr, "  pos %2d tok %6d  logits rel %.3e  KL %.2e  argmax %6d ref %6d %s\n", pos, tok, err / mx, kl, id, rid, id == rid ? "" : "MISMATCH");
            tok = pos + 1 < (int)ids.size() ? ids[pos + 1] : id;
        }
        // F16 must be numerically identical; int8 is judged by distribution drift (KL) and greedy agreement
        bool ok = pos > 0 && bad == 0 && (f16 ? worst < 1e-4 : worst_kl < 0.02);
        fprintf(stderr, "[check] %s, %d positions, worst logits rel %.3e, worst KL %.2e, argmax mismatches %d -> %s\n",
                f16 ? "F16" : "int8", pos, worst, worst_kl, bad, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    if ((int)ids.size() + n > ctx) { fprintf(stderr, "prompt + n exceeds --ctx %d\n", ctx); return 2; }
    double t0 = now();
    int pos = 0, tok = -1;
    for (; pos < (int)ids.size(); pos++) tok = eng.step(ids[pos], pos, pos + 1 == (int)ids.size());
    double t1 = now();
    long long h0 = eng.hits, m0 = eng.misses; double l0 = eng.load_s;
    eng.prof = prof;
    int gen = 0;
    for (; gen < n; gen++) {
        printf("%d\n", tok); fflush(stdout);
        if (std::find(stop.begin(), stop.end(), tok) != stop.end()) { gen++; break; }
        if (gen + 1 == n) { gen++; break; }
        tok = eng.step(tok, pos++, true);
    }
    double t2 = now();
    int dec = std::max(gen - 1, 1);  // decode steps run after the prompt
    long long dh = eng.hits - h0, dm = eng.misses - m0;
    fprintf(stderr, "[prompt] %zu tok in %.2f s (%.1f tok/s, token-by-token)\n", ids.size(), t1 - t0, ids.size() / (t1 - t0));
    fprintf(stderr, "[decode] %d tok in %.2f s = %.2f tok/s | expert hit %.1f%% (%lld miss, %.2f s loading = %.1f ms/tok)\n",
            dec, t2 - t1, dec / (t2 - t1), 100.0 * dh / std::max(dh + dm, 1LL), dm, eng.load_s - l0, 1e3 * (eng.load_s - l0) / dec);
    if (prof) {
        const char* nm[] = {"attention", "deltanet", "router+sync", "expert load", "experts", "shared expert", "lm_head+argmax"};
        double tot = 0; for (int i = 0; i < 7; i++) tot += eng.tp[i];
        for (int i = 0; i < 7; i++) fprintf(stderr, "  %-15s %7.2f ms/tok  %5.1f%%\n", nm[i], 1e3 * eng.tp[i] / dec, 100 * eng.tp[i] / tot);
    }
    return 0;
}
