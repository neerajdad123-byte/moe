// MoEx v2 decode engine. Per MoE layer: resident experts run on the GPU, misses run on the CPU
// at the same time, and missed experts are copied into VRAM slots in the background (admission).
//   moex.exe --ids 760,6511,314 [--n 32] [--ctx 2048] [--stop 248044,248046] [--profile]
//   moex.exe --check ref/golden          logits vs the PyTorch reference at every golden position
//   --f16                                 keep dense weights F16 (exact); default is int8 per-row
//   moex.exe --batch ids.txt [--trace routes.bin]   one prompt (comma ids) per line; expert cache stays warm
//     trace record: int16[12] = prompt, pos, layer, phase (0 prompt / 1 decode), 8 routed expert ids
// Generated token ids go to stdout one per line; stats go to stderr. Tokenizing is moex.py's job.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "st.h"
#include "kernels.cuh"
#include "cpu_expert.h"

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
    cudaStream_t cs, cp;  // compute, copy (both non-blocking: the legacy default stream would serialize them)
    // expert slot cache (LRU); a slot is usable once ready[] (its admission copy has finished)
    uint8_t* arena = nullptr;
    int nslots = 0;
    std::vector<int> slot_of, owner;
    std::vector<char> ready, inflight;
    std::vector<uint64_t> last;
    uint64_t clock = 0;
    long long hits = 0, misses = 0, admits = 0;
    double cpu_s = 0;
    std::vector<const uint8_t*> eq;  // mmapped expert blobs [NL*E]
    std::vector<const float*> es;    // and their scales
    // CPU side of the MoE
    CpuExperts cpu{getenv("MOEX_CPU_THREADS") ? atoi(getenv("MOEX_CPU_THREADS")) : 8};
    float *h_h, *w_h, *y_h, *hw_h, *ycpu;  // pinned host / device
    int* ids_h;
    const uint8_t** hs_h;
    float* hw_d;
    // admission: loader thread copies mmap -> pinned stage -> slot on the copy stream
    static constexpr int NSTAGE = 32;
    struct Stage { uint8_t* buf; cudaEvent_t after_compute, done; int slot = -1, key = -1; };
    Stage stage[NSTAGE];
    std::vector<int> free_st;                 // main thread only
    std::deque<int> todo, flying;             // guarded by mu
    std::mutex mu;
    std::thread loader;
    bool stop_loader = false;
    FILE* trace = nullptr;
    int trace_prompt = 0, trace_phase = 0;
    // profile (synchronizing, so it perturbs the speed it measures)
    bool prof = false;
    double tp[8] = {};

    explicit Engine(const char* root, int ctx_) : C(root), ctx(ctx_) {}
    ~Engine() {
        if (loader.joinable()) { { std::lock_guard<std::mutex> g(mu); stop_loader = true; } loader.join(); }
    }

    void loader_main() {
        while (true) {
            int i = -1;
            {
                std::lock_guard<std::mutex> g(mu);
                if (stop_loader) return;
                if (!todo.empty()) { i = todo.front(); todo.pop_front(); }
            }
            if (i < 0) { std::this_thread::sleep_for(std::chrono::microseconds(50)); continue; }
            Stage& st = stage[i];
            memcpy(st.buf, eq[st.key], EXPERT_Q_BYTES);  // may page-fault from SSD: this thread absorbs it
            memcpy(st.buf + EXPERT_Q_BYTES, es[st.key], EXPERT_BYTES - EXPERT_Q_BYTES);
            CK(cudaStreamWaitEvent(cp, st.after_compute));  // nothing still reads the victim slot
            CK(cudaMemcpyAsync(arena + (size_t)st.slot * EXPERT_BYTES, st.buf, EXPERT_BYTES, cudaMemcpyHostToDevice, cp));
            CK(cudaEventRecord(st.done, cp));
            std::lock_guard<std::mutex> g(mu);
            flying.push_back(i);
        }
    }

    // A CPU miss on a mapped page this process hasn't touched costs ~2.4 ms vs ~0.06 ms touched
    // (measured, K14). So: read every expert into RAM up front, fault each page into the working
    // set once, and ask Windows to keep the working set that large.
    void pin_experts_in_ram() {
        double t0 = now();
        std::vector<WIN32_MEMORY_RANGE_ENTRY> r;
        size_t total = 0;
        for (int k = 0; k < NL * E; k++) {
            r.push_back({(void*)eq[k], EXPERT_Q_BYTES});
            r.push_back({(void*)es[k], EXPERT_BYTES - EXPERT_Q_BYTES});
            total += EXPERT_BYTES;
        }
        // ask for the largest minimum working set Windows will grant, in 1 GB steps
        SIZE_T got = 0;
        for (SIZE_T mn = total + (1ull << 30); mn > (2ull << 30) && !got; mn -= 1ull << 30)
            if (SetProcessWorkingSetSizeEx(GetCurrentProcess(), mn, mn + (2ull << 30), QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) got = mn;
        // Fill the file cache with big sequential reads (4 KB page faults on a cold map read at random-I/O
        // speed), then fault the now-cached pages into the working set; 4 threads, one layer file at a time.
        std::atomic<int> next{0};
        std::atomic<uint64_t> sum{0};
        auto work = [&] {
            std::vector<uint8_t> buf(16 << 20);
            for (int l; (l = next.fetch_add(1)) < NL;) {
                HANDLE f = CreateFileA(C.at("model.layers." + std::to_string(l) + ".mlp.experts.0.qs").file.c_str(), GENERIC_READ,
                                       FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                DWORD got_b = 0;
                while (f != INVALID_HANDLE_VALUE && ReadFile(f, buf.data(), (DWORD)buf.size(), &got_b, nullptr) && got_b) {}
                if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
                uint64_t s = 0;
                for (size_t i = (size_t)l * E * 2; i < (size_t)(l + 1) * E * 2; i++)
                    for (size_t o = 0; o < r[i].NumberOfBytes; o += 4096) s += ((const uint8_t*)r[i].VirtualAddress)[o];
                sum += s;
            }
        };
        std::vector<std::thread> ths;
        for (int i = 0; i < 4; i++) ths.emplace_back(work);
        for (auto& t : ths) t.join();
        static volatile uint64_t sink; sink = sum;
        fprintf(stderr, "[ram] %.1f GB of experts faulted in, %.1f s; min working set %.1f GB%s\n", total / 1e9, now() - t0,
                got / 1e9, got >= total ? "" : " (below the expert total: some CPU misses will page-fault)");
    }

    // Publish finished admissions (main thread).
    void poll_admissions() {
        std::lock_guard<std::mutex> g(mu);
        for (size_t n = flying.size(); n--;) {
            int i = flying.front(); flying.pop_front();
            if (cudaEventQuery(stage[i].done) == cudaSuccess) {
                Stage& st = stage[i];
                if (owner[st.slot] == st.key) ready[st.slot] = 1;
                inflight[st.slot] = 0;
                free_st.push_back(i);
            } else flying.push_back(i);
        }
    }

    // Start copying expert `key` into the LRU victim slot; skipped when all stages are busy.
    void admit(int key) {
        if (free_st.empty()) return;
        int s = -1; uint64_t best = ~0ull;
        for (int i = 0; i < nslots; i++) if (!inflight[i] && last[i] < best) { best = last[i]; s = i; }
        if (s < 0) return;
        if (owner[s] >= 0) slot_of[owner[s]] = -1;
        owner[s] = key; slot_of[key] = s; ready[s] = 0; inflight[s] = 1; last[s] = clock;
        int i = free_st.back(); free_st.pop_back();
        stage[i].slot = s; stage[i].key = key;
        CK(cudaEventRecord(stage[i].after_compute, cs));
        admits++;
        std::lock_guard<std::mutex> g(mu);
        todo.push_back(i);
    }

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
        CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking));
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
        ready.assign(nslots, 0); inflight.assign(nslots, 0);
        eq.resize(NL * E); es.resize(NL * E);
        for (int l = 0; l < NL; l++) for (int e = 0; e < E; e++) {
            std::string ep = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
            eq[l * E + e] = C.map(ep + "merged_weight");
            es[l * E + e] = (const float*)C.map(ep + "qs");
        }
        CK(cudaMallocHost(&h_h, D * 4)); CK(cudaMallocHost(&y_h, D * 4)); CK(cudaMallocHost(&w_h, TOPK * 4));
        CK(cudaMallocHost(&hw_h, TOPK * 4)); CK(cudaMallocHost(&ids_h, TOPK * 4)); CK(cudaMallocHost(&hs_h, TOPK * sizeof(void*)));
        CK(cudaMalloc(&hw_d, TOPK * 4)); CK(cudaMalloc(&ycpu, D * 4));
        for (int i = 0; i < NSTAGE; i++) {
            CK(cudaMallocHost(&stage[i].buf, EXPERT_BYTES));
            CK(cudaEventCreateWithFlags(&stage[i].after_compute, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stage[i].done, cudaEventDisableTiming));
            free_st.push_back(i);
        }
        loader = std::thread([this] { loader_main(); });
        pin_experts_in_ram();
        fprintf(stderr, "[load] %.1f s, dense %s, %d expert slots (%.2f GB, %.1f%% of %d experts), ctx %d\n",
                now() - t0, f16 ? "F16" : "int8", nslots, nslots * (double)EXPERT_BYTES / 1e9, 100.0 * nslots / (NL * E), NL * E, ctx);
    }

    // New conversation: recurrent state back to zero (KV rows are overwritten position by position).
    void reset() {
        for (int l = 0; l < NL; l++) if (!is_attn(l)) {
            CK(cudaMemsetAsync(L[l].ring, 0, (size_t)DC * (CONVK - 1) * 4, cs));
            CK(cudaMemsetAsync(L[l].S, 0, (size_t)DVH * DNK * DNV * 4, cs));
        }
    }

    void mark(int i, double& t) { if (prof) { CK(cudaStreamSynchronize(cs)); double n = now(); tp[i] += n - t; t = n; } }

    // One token at position `pos`. If want_logits, logits[] is filled and the greedy id returned.
    int step(int tok, int pos, bool want_logits) {
        double t = now();
        for (int i = 0; i < D; i++) h_h[i] = __half2float(embed[(size_t)tok * D + i]);
        CK(cudaMemcpyAsync(x, h_h, D * 4, cudaMemcpyHostToDevice, cs));
        const float* pending = nullptr;  // previous MoE output, added to x by the next norm
        for (int l = 0; l < NL; l++) {
            LayerW& w = L[l];
            rmsnorm(x, pending, w.in_ln, h, D, EPS, 1, cs);
            if (is_attn(l)) {
                gemv(w.q, h, q, D, AH * 2 * AHD, cs); gemv(w.k, h, k, D, AKV * AHD, cs); gemv(w.v, h, v, D, AKV * AHD, cs);
                attn_decode(q, k, v, w.qn, w.kn, qo, w.K, w.V, ag, AH, AKV, pos, ctx, ROT, THETA, EPS, cs);
                gemv(w.o, ag, tmp, AH * AHD, D, cs);
                mark(0, t);
            } else {
                gemv(w.qkv, h, big, D, DC, cs); gemv(w.z, h, z, D, DVH * DNV, cs);
                gemv_f16(w.a, h, a, D, DVH, cs); gemv_f16(w.b, h, b, D, DVH, cs);
                deltanet_core(big, z, a, b, w.conv, w.alog, w.dtb, w.nw, w.ring, w.S, conv, o, DC, CONVK, DVH, DKH, EPS, cs);
                gemv(w.out, o, tmp, DVH * DNV, D, cs);
                mark(1, t);
            }
            rmsnorm(x, tmp, w.post_ln, h, D, EPS, 1, cs);
            gemv_f16(w.router, h, scr, D, E, cs);
            router_topk(scr, E, TOPK, false, ridx, rw, cs);
            CK(cudaMemcpyAsync(ids_h, ridx, TOPK * 4, cudaMemcpyDeviceToHost, cs));
            CK(cudaMemcpyAsync(w_h, rw, TOPK * 4, cudaMemcpyDeviceToHost, cs));
            CK(cudaMemcpyAsync(h_h, h, D * 4, cudaMemcpyDeviceToHost, cs));
            CK(cudaStreamSynchronize(cs));  // per-layer sync: placement needs the routing
            mark(2, t);
            if (trace) {
                int16_t rec[12] = {(int16_t)trace_prompt, (int16_t)pos, (int16_t)l, (int16_t)trace_phase};
                for (int j = 0; j < TOPK; j++) rec[4 + j] = (int16_t)ids_h[j];
                fwrite(rec, sizeof rec, 1, trace);
            }
            poll_admissions();
            clock++;
            int nh = 0, nm = 0, mkey[TOPK];
            const uint8_t* mq[TOPK]; const float* msc[TOPK]; float mw[TOPK];
            for (int j = 0; j < TOPK; j++) {
                int key = l * E + ids_h[j], sl = slot_of[key];
                if (sl >= 0) last[sl] = clock;  // also protects in-flight siblings from eviction
                if (sl >= 0 && ready[sl]) { hs_h[nh] = arena + (size_t)sl * EXPERT_BYTES; hw_h[nh++] = w_h[j]; hits++; }
                else { mq[nm] = eq[key]; msc[nm] = es[key]; mw[nm] = w_h[j]; mkey[nm++] = sl < 0 ? key : -1; misses++; }
            }
            // GPU: resident experts + shared expert (async)
            if (nh) {
                CK(cudaMemcpyAsync(slots_d, hs_h, nh * sizeof(void*), cudaMemcpyHostToDevice, cs));
                CK(cudaMemcpyAsync(hw_d, hw_h, nh * 4, cudaMemcpyHostToDevice, cs));
                experts_forward(slots_d, hw_d, nh, h, act, moe, cs);
            } else CK(cudaMemsetAsync(moe, 0, D * 4, cs));
            shared_expert_add(w.sh_g, w.sh_u, w.sh_d, w.sh_gate, h, moe, scr, D, SI, cs);
            // CPU: misses, overlapped with the GPU work above
            if (nm) {
                double c0 = now();
                cpu.run(nm, mq, msc, mw, h_h, y_h);
                cpu_s += now() - c0;
                CK(cudaMemcpyAsync(ycpu, y_h, D * 4, cudaMemcpyHostToDevice, cs));
                add_k<<<(D + 255) / 256, 256, 0, cs>>>(moe, ycpu, D);
                for (int j = 0; j < nm; j++) if (mkey[j] >= 0) admit(mkey[j]);
            }
            mark(3, t);
            pending = moe;
        }
        // the last MoE output is folded into x here, or by the next token's first norm
        rmsnorm(x, pending, norm, h, D, EPS, 1, cs);
        if (!want_logits) return -1;
        gemv(lm_head, h, logits, D, V, cs);
        argmax(logits, V, amv, ams, tok_d, cs);
        CK(cudaMemcpyAsync(ids_h, tok_d, 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaStreamSynchronize(cs));
        mark(4, t);
        return ids_h[0];
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
    std::string check, batch, trace_path;
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
        else if (s == "--batch") batch = next();
        else if (s == "--trace") trace_path = next();
        else { fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }
    if (!check.empty()) {  // ids come from the golden run's meta.json
        FILE* f = fopen((check + "/meta.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s/meta.json\n", check.c_str()); return 1; }
        std::string m(1 << 16, '\0'); m.resize(fread(m.data(), 1, m.size(), f)); fclose(f);
        size_t a = m.find('[', m.find("\"ids\"")) + 1, b = m.find(']', a);
        ids = parse_ids(m.substr(a, b - a));
    }
    if (ids.empty() && batch.empty()) { fprintf(stderr, "usage: moex --ids 1,2,3 [--n N] | --batch ids.txt | --check ref/golden\n"); return 2; }

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

    struct Stats { int dec = 0; double sec = 0; long long hits = 0, misses = 0, admits = 0; double cpu = 0; };
    // Prompt token-by-token, then up to n greedy tokens. Returns decode-phase stats.
    auto generate = [&](const std::vector<int>& p, std::vector<int>& out, bool stream) {
        Stats st;
        if ((int)p.size() + n > ctx) { fprintf(stderr, "prompt + n exceeds --ctx %d\n", ctx); exit(2); }
        eng.trace_phase = 0;
        int pos = 0, tok = -1;
        for (; pos < (int)p.size(); pos++) tok = eng.step(p[pos], pos, pos + 1 == (int)p.size());
        long long h0 = eng.hits, m0 = eng.misses, a0 = eng.admits; double c0 = eng.cpu_s, t1 = now();
        eng.trace_phase = 1;
        for (int gen = 0; gen < n; gen++) {
            out.push_back(tok);
            if (stream) { printf("%d\n", tok); fflush(stdout); }
            if (std::find(stop.begin(), stop.end(), tok) != stop.end() || gen + 1 == n) break;
            tok = eng.step(tok, pos++, true);
            st.dec++;
        }
        st.sec = now() - t1; st.hits = eng.hits - h0; st.misses = eng.misses - m0; st.admits = eng.admits - a0; st.cpu = eng.cpu_s - c0;
        return st;
    };
    auto report = [&](const char* tag, const Stats& st) {
        int dec = std::max(st.dec, 1);
        fprintf(stderr, "[%s] %d tok in %.2f s = %.2f tok/s | expert hit %.1f%% | %.1f misses/tok on CPU = %.1f ms/tok | %lld admitted\n",
                tag, st.dec, st.sec, st.dec / std::max(st.sec, 1e-9), 100.0 * st.hits / std::max(st.hits + st.misses, 1LL),
                (double)st.misses / dec, 1e3 * st.cpu / dec, st.admits);
    };
    if (!trace_path.empty() && !(eng.trace = fopen(trace_path.c_str(), "wb"))) { fprintf(stderr, "cannot write %s\n", trace_path.c_str()); return 1; }

    if (!batch.empty()) {
        FILE* f = fopen(batch.c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s\n", batch.c_str()); return 1; }
        std::vector<std::vector<int>> prompts;
        char line[1 << 16];
        while (fgets(line, sizeof line, f)) { auto v = parse_ids(line); if (!v.empty()) prompts.push_back(v); }
        fclose(f);
        Stats tot;
        for (size_t i = 0; i < prompts.size(); i++) {
            eng.reset();
            eng.trace_prompt = (int)i;
            std::vector<int> out;
            Stats st = generate(prompts[i], out, false);
            for (size_t j = 0; j < out.size(); j++) printf(j ? ",%d" : "%d", out[j]);
            printf("\n"); fflush(stdout);
            char tag[32]; snprintf(tag, sizeof tag, "prompt %zu", i);
            report(tag, st);
            tot.dec += st.dec; tot.sec += st.sec; tot.hits += st.hits; tot.misses += st.misses; tot.admits += st.admits; tot.cpu += st.cpu;
        }
        report("batch total", tot);
        if (eng.trace) fclose(eng.trace);
        return 0;
    }

    eng.prof = prof;
    std::vector<int> out;
    Stats st = generate(ids, out, true);
    report("decode", st);
    if (prof) {
        const char* nm[] = {"attention", "deltanet", "router+sync", "moe (gpu+cpu)", "lm_head+argmax"};
        double tot = 0; for (int i = 0; i < 5; i++) tot += eng.tp[i];
        for (int i = 0; i < 5; i++) fprintf(stderr, "  %-15s %7.2f ms/tok  %5.1f%%\n", nm[i], 1e3 * eng.tp[i] / std::max(st.dec, 1), 100 * eng.tp[i] / tot);
    }
    if (eng.trace) fclose(eng.trace);
    return 0;
}
