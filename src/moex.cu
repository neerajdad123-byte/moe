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
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
static unsigned long page_faults() { PROCESS_MEMORY_COUNTERS c{}; GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c); return c.PageFaultCount; }

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Qwen3.6-35B-A3B geometry (checked against the container shapes at load).
constexpr int D = 2048, NL = 40, E = 256, TOPK = 8, SI = 512, V = 248320;
constexpr int AH = 16, AKV = 2, ROT = 64;            // attention
constexpr int DC = 8192, DVH = 32, DKH = 16, CONVK = 4;  // DeltaNet
constexpr float EPS = 1e-6f, THETA = 1e7f;
static bool is_attn(int l) { return l % 4 == 3; }

struct LayerW {
    half *in_ln, *post_ln, *router, *sh_gate, *qn, *kn, *conv, *nw, *alog, *dtb;  // small: F16 always
    // big: int8 unless --f16. Projections that share an input are one fused matrix (one launch, less tail):
    Mat sh_gu, sh_d, attn_qkv, o, dn_in, out;  // sh_gu=[gate;up], attn_qkv=[q;k;v], dn_in=[qkv;z;a;b]
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
    int *ridx, *ams;
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
    double ht[4] = {};  // host time per phase: launch (poll+flush+graph), wait for routing, CPU misses, admissions
    std::vector<const uint8_t*> eq;  // mmapped expert blobs [NL*E]
    std::vector<const float*> es;    // and their scales
    // CPU side of the MoE
    CpuExperts cpu{getenv("MOEX_CPU_THREADS") ? atoi(getenv("MOEX_CPU_THREADS")) : 8};
    // Host buffers mapped into the GPU's address space (zero-copy): kernels write the routing straight to the
    // host and read the CPU experts' result straight from it, so a layer needs no memcpy operations
    // (each small async copy costs a separate DMA op, ~10-20 us on WDDM; ~10% of the decode timeline).
    float *h_h, *w_h, *y_h;           // host views
    float *h_hd, *w_hd, *y_hd;        // device views of the same memory
    int* ids_hd;
    int* ids_h;
    float* hw_d;
    int *nh_d, *pos_d;
    cudaEvent_t ev_route;
    // Device copy of the slot table (key -> slot pointer, null = not resident), so the GPU picks its
    // hits without a host round trip. Updated in stream order on cs by table_k with values passed as
    // kernel arguments (a pinned-memory copy would read the host value at execution time, not enqueue time).
    const uint8_t** table_d;
    TableUpd upd;
    std::vector<int> new_jobs;
    void table_set(int key, const uint8_t* p) {
        if (upd.n == TableUpd::N) flush_table();
        upd.key[upd.n] = key; upd.ptr[upd.n++] = p;
    }
    void flush_table() {
        if (upd.n) { table_k<<<1, TableUpd::N, 0, cs>>>(table_d, upd); upd.n = 0; }
    }
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
        // keep them there: the minimum working set is only a hint, and prefill (which touches every expert)
        // measured 156k soft faults per 512 tokens without locks
        size_t locked = 0;
        for (auto& x : r) if (VirtualLock(x.VirtualAddress, x.NumberOfBytes)) locked += x.NumberOfBytes;
        fprintf(stderr, "[ram] %.1f GB of experts locked in the working set\n", locked / 1e9);
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
                if (owner[st.slot] == st.key) { ready[st.slot] = 1; table_set(st.key, arena + (size_t)st.slot * EXPERT_BYTES); }
                inflight[st.slot] = 0;
                free_st.push_back(i);
            } else flying.push_back(i);
        }
    }

    // Eviction score: decayed use count with a 10-token half-life (log domain). Measured on 80-prompt
    // traces (tools/sim.py): 82.0% hit vs 80.8% LRU; longer half-lives lose (tau 200: 69%).
    static constexpr double TAU = 10.0;
    std::vector<double> G = std::vector<double>(NL * E, -INFINITY);
    long long tokens = 0;
    void touch(int key) {
        double v = tokens / TAU, g = G[key];
        G[key] = g == -INFINITY ? v : std::max(g, v) + log1p(exp(-fabs(g - v)));
    }

    // Start copying expert `key` into the lowest-scoring slot not used this layer step; skipped when all stages are busy.
    void admit(int key) {
        if (free_st.empty()) return;
        int s = -1; double best = INFINITY;
        for (int i = 0; i < nslots; i++) {
            double sc = owner[i] < 0 ? -INFINITY : G[owner[i]];
            if (!inflight[i] && last[i] != clock && sc < best) { best = sc; s = i; }
        }
        if (s < 0) return;
        if (owner[s] >= 0) { slot_of[owner[s]] = -1; if (ready[s]) table_set(owner[s], nullptr); }
        owner[s] = key; slot_of[key] = s; ready[s] = 0; inflight[s] = 1; last[s] = clock;
        int i = free_st.back(); free_st.pop_back();
        stage[i].slot = s; stage[i].key = key;
        admits++;
        new_jobs.push_back(i);
    }
    // After a layer's admit() calls: evictions reach the device table, then the copies may start
    // (each waits for everything enqueued on cs so far, which includes the last reads of its victim).
    void start_admissions() {
        if (new_jobs.empty()) return;
        flush_table();
        for (int i : new_jobs) CK(cudaEventRecord(stage[i].after_compute, cs));
        std::lock_guard<std::mutex> g(mu);
        for (int i : new_jobs) todo.push_back(i);
        new_jobs.clear();
    }

    half* wup(const std::string& n, int64_t numel) {
        const StTensor& s = C.at(n);
        if ((int64_t)s.bytes != numel * 2) { fprintf(stderr, "shape mismatch %s\n", n.c_str()); exit(1); }
        half* d; CK(cudaMalloc(&d, s.bytes));
        CK(cudaMemcpy(d, C.map(n), s.bytes, cudaMemcpyHostToDevice));
        return d;
    }
    // Row-concatenation of matrices with the same input width I, as F16 or int8 + per-row scales.
    Mat wmat(std::initializer_list<std::pair<std::string, int64_t>> parts, int64_t I) {
        int64_t O = 0;
        for (auto& [n, rows] : parts) {
            if ((int64_t)C.at(n).bytes != rows * I * 2) { fprintf(stderr, "shape mismatch %s\n", n.c_str()); exit(1); }
            O += rows;
        }
        half* f = nullptr; int8_t* q = nullptr; float* sc = nullptr;
        if (f16) CK(cudaMalloc(&f, O * I * 2));
        else { CK(cudaMalloc(&q, O * I)); CK(cudaMalloc(&sc, O * 4)); }
        int64_t off = 0;
        for (auto& [n, rows] : parts) {
            if (f16) CK(cudaMemcpy(f + off * I, C.map(n), rows * I * 2, cudaMemcpyHostToDevice));
            else {  // the synchronous copy also orders qtmp's reuse after the previous quantize
                CK(cudaMemcpy(qtmp, C.map(n), rows * I * 2, cudaMemcpyHostToDevice));
                quantize_rows_i8_k<<<(int)rows, 256>>>(qtmp, q + off * I, sc + off, (int)I);
                CK(cudaGetLastError());
            }
            off += rows;
        }
        return {f, q, sc};
    }
    static float* falloc(size_t n, bool zero = false) {
        float* d; CK(cudaMalloc(&d, n * 4)); if (zero) CK(cudaMemset(d, 0, n * 4)); return d;
    }

    void load() {
        double t0 = now();
        embed = (const half*)C.map("model.embed_tokens.weight");
        if (!f16) CK(cudaMalloc(&qtmp, (size_t)V * D * 2));  // largest matrix
        lm_head = wmat({{"lm_head.weight", V}}, D);
        norm = wup("model.norm.weight", D);
        L.resize(NL);
        for (int i = 0; i < NL; i++) {
            std::string p = "model.layers." + std::to_string(i) + ".", m = p + "mlp.";
            LayerW& w = L[i];
            w.in_ln = wup(p + "input_layernorm.weight", D);
            w.post_ln = wup(p + "post_attention_layernorm.weight", D);
            w.router = wup(m + "gate.weight", (int64_t)E * D);
            w.sh_gu = wmat({{m + "shared_expert.gate_proj.weight", SI}, {m + "shared_expert.up_proj.weight", SI}}, D);
            w.sh_d = wmat({{m + "shared_expert.down_proj.weight", D}}, SI);
            w.sh_gate = wup(m + "shared_expert_gate.weight", D);
            if (is_attn(i)) {
                std::string a = p + "self_attn.";
                w.attn_qkv = wmat({{a + "q_proj.weight", AH * 2 * AHD}, {a + "k_proj.weight", AKV * AHD}, {a + "v_proj.weight", AKV * AHD}}, D);
                w.o = wmat({{a + "o_proj.weight", D}}, AH * AHD);
                w.qn = wup(a + "q_norm.weight", AHD);
                w.kn = wup(a + "k_norm.weight", AHD);
                w.K = falloc((size_t)AKV * ctx * AHD);
                w.V = falloc((size_t)AKV * ctx * AHD);
            } else {
                std::string a = p + "linear_attn.";
                w.dn_in = wmat({{a + "in_proj_qkv.weight", DC}, {a + "in_proj_z.weight", DVH * DNV},
                                {a + "in_proj_a.weight", DVH}, {a + "in_proj_b.weight", DVH}}, D);
                w.out = wmat({{a + "out_proj.weight", D}}, DVH * DNV);
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
        q = falloc(AH * 2 * AHD + 2 * AKV * AHD); k = q + AH * 2 * AHD; v = k + AKV * AHD; qo = falloc(AH * AHD); ag = falloc(AH * AHD);
        big = falloc(DC + DVH * DNV + 2 * DVH); z = big + DC; a = z + DVH * DNV; b = a + DVH;
        conv = falloc(DC); o = falloc(DVH * DNV);
        act = falloc(TOPK * EI); scr = falloc(2 * SI + D + 1); logits = falloc(V); rw = falloc(TOPK);
        amv = falloc(ARGMAX_BLOCKS + 1);
        CK(cudaMalloc(&ridx, TOPK * 4)); CK(cudaMalloc(&ams, ARGMAX_BLOCKS * 4));
        CK(cudaMalloc(&slots_d, TOPK * sizeof(void*)));
        if (ctx * 4 > 48 * 1024)
            CK(cudaFuncSetAttribute(attn_decode_k, cudaFuncAttributeMaxDynamicSharedMemorySize, ctx * 4));
        alloc_prefill();
        eq.resize(NL * E); es.resize(NL * E);
        for (int l = 0; l < NL; l++) for (int e = 0; e < E; e++) {
            std::string ep = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + ".";
            eq[l * E + e] = C.map(ep + "merged_weight");
            es[l * E + e] = (const float*)C.map(ep + "qs");
        }
        pin_experts_in_ram();
        // every byte of VRAM left (minus headroom for the driver) becomes expert slots
        size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = 320ull << 20;
        nslots = (int)((fr - std::min(fr, reserve)) / EXPERT_BYTES);
        if (nslots < 2 * TOPK) { fprintf(stderr, "only %d expert slots fit; need >= %d\n", nslots, 2 * TOPK); exit(1); }
        CK(cudaMalloc(&arena, (size_t)nslots * EXPERT_BYTES));
        slot_of.assign(NL * E, -1); owner.assign(nslots, -1); last.assign(nslots, 0);
        ready.assign(nslots, 0); inflight.assign(nslots, 0);
        auto mapped = [](auto** hp, auto** dp, size_t bytes) {
            CK(cudaHostAlloc((void**)hp, bytes, cudaHostAllocMapped));
            CK(cudaHostGetDevicePointer((void**)dp, *hp, 0));
        };
        mapped(&h_h, &h_hd, D * 4); mapped(&y_h, &y_hd, D * 4); mapped(&w_h, &w_hd, TOPK * 4); mapped(&ids_h, &ids_hd, TOPK * 4);
        CK(cudaMalloc(&hw_d, TOPK * 4)); CK(cudaMalloc(&nh_d, 4));
        CK(cudaMalloc(&table_d, NL * E * sizeof(void*))); CK(cudaMemset(table_d, 0, NL * E * sizeof(void*)));
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
        CK(cudaMalloc(&pos_d, 4));
        for (int i = 0; i < NSTAGE; i++) {
            CK(cudaMallocHost(&stage[i].buf, EXPERT_BYTES));
            CK(cudaEventCreateWithFlags(&stage[i].after_compute, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stage[i].done, cudaEventDisableTiming));
            free_st.push_back(i);
        }
        loader = std::thread([this] { loader_main(); });
        capture_graphs();
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

    // All of layer l's GPU work: mixer, post norm, router (+ routing to the host, ev_route), then the
    // resident experts and the shared expert. Host-independent, so it is captured once as a CUDA graph:
    // ~18 launches at ~7 us each on WDDM become one graph launch.
    void enqueue_layer(int l) {
        LayerW& w = L[l];
        rmsnorm(x, l ? moe : nullptr, w.in_ln, h, D, EPS, 1, cs);  // folds in the previous layer's MoE output
        if (is_attn(l)) {
            gemv(w.attn_qkv, h, q, D, AH * 2 * AHD + 2 * AKV * AHD, cs);  // -> q, k, v
            attn_decode(q, k, v, w.qn, w.kn, qo, w.K, w.V, ag, AH, AKV, pos_d, ctx, ROT, THETA, EPS, 1, 0, cs);
            gemv(w.o, ag, tmp, AH * AHD, D, cs);
        } else {
            gemv(w.dn_in, h, big, D, DC + DVH * DNV + 2 * DVH, cs);  // -> qkv, z, a, b
            deltanet_core(big, z, a, b, DC + DVH * DNV + 2 * DVH, w.conv, w.alog, w.dtb, w.nw, w.ring, w.S, conv, o, DC, CONVK, DVH, DKH, EPS, 1, cs);
            gemv(w.out, o, tmp, DVH * DNV, D, cs);
        }
        rmsnorm(x, tmp, w.post_ln, h, D, EPS, 1, cs);
        gemv_f16(w.router, h, scr, D, E, cs);
        router_topk(scr, E, TOPK, false, ridx, rw, cs);
        export_route_k<<<(D + 255) / 256, 256, 0, cs>>>(h, ridx, rw, h_hd, ids_hd, w_hd, D, TOPK);
        CK(cudaEventRecordWithFlags(ev_route, cs, cudaEventRecordExternal));  // host-visible even from a graph
        // hits from the device slot table, then resident experts + shared expert, no host in the loop
        route_hits_k<<<1, 32, 0, cs>>>(ridx, rw, TOPK, l * E, table_d, slots_d, hw_d, nh_d);
        experts_forward(slots_d, hw_d, TOPK, h, act, moe, cs, nh_d);
        shared_expert_add(w.sh_gu, w.sh_d, w.sh_gate, h, moe, scr, D, SI, cs);
    }

    std::vector<cudaGraphExec_t> graphs;
    void capture_graphs() {
        graphs.resize(NL);
        for (int l = 0; l < NL; l++) {
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
            enqueue_layer(l);
            CK(cudaStreamEndCapture(cs, &g));
            CK(cudaGraphInstantiate(&graphs[l], g, 0));
            CK(cudaGraphDestroy(g));
        }
    }

    // One token at position `pos`. If want_logits, logits[] is filled and the greedy id returned.
    // ---- prefill: the prompt in chunks of CH tokens; every weight is read once per chunk ----
    static constexpr int CH = 256;
    float *Xc, *hc, *tmpc, *moec, *qkvc, *bigc, *convc, *oc, *qoc, *agc, *lgc, *rwc, *actp, *ypair, *shgu, *sha, *shs, *shl;
    int* ridxc;
    float *rows_h, *rows_d, *rwc_h, *rwc_d, *ycpu_h, *ycpu_d;  // mapped host memory (zero-copy)
    int *ridxc_h, *ridxc_d;
    PairItem *items_h, *items_d;
    Pair *pairs_h, *pairs_d;
    std::vector<float> ypair_cpu;
    std::vector<std::vector<Pair>> per = std::vector<std::vector<Pair>>(E);
    long long streamed_n = 0;
    // Prefill streams non-resident experts to a VRAM staging area: worker threads copy them from the file map into
    // a pinned host ring, and each batch goes over PCIe as one DMA, double-buffered. (Page-locking the map itself
    // for direct DMA was measured: 12.2 GB/s, but on WDDM it comes out of the GPU's shared-memory budget, and
    // locking/unlocking layers evicted the pages from the working set: decode CPU misses went 3 -> 79 ms/token.)
    static constexpr int SB = 24;   // experts per staging batch; 2 * SB staging slots
    static constexpr int STREAM_MIN = 2;  // an expert used by one token is cheaper on the CPU than over PCIe
    uint8_t* stage_p = nullptr;   // VRAM, 2 * SB slots
    uint8_t* hstage = nullptr;    // pinned host, 2 * SB slots
    CpuPool cpool{4};             // staging copies, concurrent with the CPU experts on cpu.pool
    cudaEvent_t ev_hfree[2];
    cudaStream_t cpp;
    cudaEvent_t ev_copied[2], ev_free[2];
    template <class T> static void mapped_alloc(T** hp, T** dp, size_t n) {
        CK(cudaHostAlloc((void**)hp, n * sizeof(T), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**)dp, *hp, 0));
    }
    void alloc_prefill() {
        const int LQ = AH * 2 * AHD + 2 * AKV * AHD, LB = DC + DVH * DNV + 2 * DVH;
        Xc = falloc((size_t)CH * D); hc = falloc((size_t)CH * D); tmpc = falloc((size_t)CH * D); moec = falloc((size_t)CH * D);
        qkvc = falloc((size_t)CH * LQ); bigc = falloc((size_t)CH * LB); convc = falloc((size_t)CH * DC); oc = falloc((size_t)CH * DVH * DNV);
        qoc = falloc((size_t)CH * AH * AHD); agc = falloc((size_t)CH * AH * AHD); lgc = falloc((size_t)CH * E); rwc = falloc((size_t)CH * TOPK);
        actp = falloc((size_t)CH * TOPK * EI); ypair = falloc((size_t)CH * TOPK * D);
        shgu = falloc((size_t)CH * 2 * SI); sha = falloc((size_t)CH * SI); shs = falloc((size_t)CH * D); shl = falloc(CH);
        CK(cudaMalloc(&ridxc, CH * TOPK * 4));
        mapped_alloc(&rows_h, &rows_d, (size_t)CH * D); mapped_alloc(&rwc_h, &rwc_d, (size_t)CH * TOPK);
        mapped_alloc(&ycpu_h, &ycpu_d, (size_t)CH * D); mapped_alloc(&ridxc_h, &ridxc_d, (size_t)CH * TOPK);
        mapped_alloc(&items_h, &items_d, (size_t)E); mapped_alloc(&pairs_h, &pairs_d, (size_t)CH * TOPK);
        ypair_cpu.resize((size_t)CH * TOPK * D);
        CK(cudaMalloc(&stage_p, (size_t)2 * SB * EXPERT_BYTES));
        CK(cudaMallocHost(&hstage, (size_t)2 * SB * EXPERT_BYTES));
        CK(cudaStreamCreateWithFlags(&cpp, cudaStreamNonBlocking));
        for (int i = 0; i < 2; i++) {
            CK(cudaEventCreateWithFlags(&ev_copied[i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_free[i], cudaEventDisableTiming));
            CK(cudaEventRecord(ev_free[i], cs));
            CK(cudaEventCreateWithFlags(&ev_hfree[i], cudaEventDisableTiming));
            CK(cudaEventRecord(ev_hfree[i], cpp));
        }
    }

    // Runs prompt tokens ids[0..n) at positions pos0.. and returns the greedy next token (logits[] filled).
    // Same math as step() except dense projections (fp16 activations on tensor cores) and summation order.
    int prefill(const int* ids, int n, int pos0) {
        const int LQ = AH * 2 * AHD + 2 * AKV * AHD, LB = DC + DVH * DNV + 2 * DVH;
        int next = -1;
        double pt0 = now(), pt[5] = {};
        unsigned long pf0 = page_faults();
        long long sn0 = streamed_n;
        for (int c0 = 0; c0 < n; c0 += CH) {
            const int T = std::min(CH, n - c0);
            tokens += T;
            for (int t = 0; t < T; t++)
                for (int i = 0; i < D; i++) rows_h[(size_t)t * D + i] = __half2float(embed[(size_t)ids[c0 + t] * D + i]);
            copy_k<<<(T * D + 255) / 256, 256, 0, cs>>>(Xc, rows_d, T * D);
            set_int_k<<<1, 1, 0, cs>>>(pos_d, pos0 + c0);
            for (int l = 0; l < NL; l++) {
                LayerW& w = L[l];
                rmsnorm(Xc, l ? moec : nullptr, w.in_ln, hc, D, EPS, T, cs);
                if (is_attn(l)) {
                    gemm(w.attn_qkv, hc, qkvc, T, D, LQ, cs);
                    attn_decode(qkvc, qkvc + AH * 2 * AHD, qkvc + AH * 2 * AHD + AKV * AHD, w.qn, w.kn, qoc, w.K, w.V, agc,
                                AH, AKV, pos_d, ctx, ROT, THETA, EPS, T, LQ, cs);
                    gemm(w.o, agc, tmpc, T, AH * AHD, D, cs);
                } else {
                    gemm(w.dn_in, hc, bigc, T, D, LB, cs);
                    deltanet_core(bigc, bigc + DC, bigc + DC + DVH * DNV, bigc + DC + DVH * DNV + DVH, LB, w.conv, w.alog, w.dtb, w.nw,
                                  w.ring, w.S, convc, oc, DC, CONVK, DVH, DKH, EPS, T, cs);
                    gemm(w.out, oc, tmpc, T, DVH * DNV, D, cs);
                }
                rmsnorm(Xc, tmpc, w.post_ln, hc, D, EPS, T, cs);
                for (int t = 0; t < T; t++) gemv_f16(w.router, hc + (size_t)t * D, lgc + (size_t)t * E, D, E, cs);  // fp32 routing, as decode
                router_topk(lgc, E, TOPK, false, ridxc, rwc, cs, T);
                export_route_k<<<(T * D + 255) / 256, 256, 0, cs>>>(hc, ridxc, rwc, rows_d, ridxc_d, rwc_d, T * D, T * TOPK);
                double w0 = now();
                CK(cudaStreamSynchronize(cs));
                pt[0] += now() - w0;
                // group the chunk's (token, expert) pairs by expert: resident -> GPU, non-resident with
                // >= STREAM_MIN tokens -> streamed to the GPU, the rest -> CPU
                for (auto& v : per) v.clear();
                for (int t = 0; t < T; t++)
                    for (int r = 0; r < TOPK; r++) {
                        int e = ridxc_h[t * TOPK + r];
                        per[e].push_back({t, t * TOPK + r, rwc_h[t * TOPK + r]});
                        touch(l * E + e);
                    }
                int nitems = 0, npairs = 0, job_of[E];
                std::fill(job_of, job_of + E, -1);
                std::vector<CpuExperts::Job> jobs;
                std::vector<int> streamed;
                auto add_item = [&](int e, const uint8_t* slot) {
                    items_h[nitems] = {slot, npairs, (int)per[e].size()};
                    for (auto& pr : per[e]) pairs_h[npairs++] = pr;
                    nitems++;
                };
                for (int e = 0; e < E; e++) {
                    if (per[e].empty()) continue;
                    int key = l * E + e, sl = slot_of[key];
                    if (sl >= 0 && ready[sl]) { add_item(e, arena + (size_t)sl * EXPERT_BYTES); hits += per[e].size(); continue; }
                    misses += per[e].size();
                    if ((int)per[e].size() >= STREAM_MIN) { streamed.push_back(e); continue; }
                    job_of[e] = (int)jobs.size();
                    jobs.push_back({eq[key], es[key], {}, {}, {}});
                    for (auto& pr : per[e]) { jobs.back().toks.push_back(pr.t); jobs.back().pids.push_back(pr.p); jobs.back().ws.push_back(pr.w); }
                }
                // GPU: resident experts and the shared expert first (all async)
                CK(cudaMemsetAsync(ypair, 0, (size_t)T * TOPK * D * 4, cs));
                const int nres = nitems;
                pairs_forward(items_d, nres, pairs_d, hc, actp, ypair, cs);
                gemm(w.sh_gu, hc, shgu, T, D, 2 * SI, cs);
                silu_mul_rows_k<<<dim3((SI + 255) / 256, T), 256, 0, cs>>>(shgu, sha, SI);
                gemm(w.sh_d, sha, shs, T, SI, D, cs);
                for (int t = 0; t < T; t++) gemv_f16(w.sh_gate, hc + (size_t)t * D, shl + t, D, 1, cs);
                // streamed experts: items point at their staging slot (batch bi uses half bi&1)
                for (int k = 0; k < (int)streamed.size(); k++)
                    add_item(streamed[k], stage_p + ((size_t)((k / SB) & 1) * SB + k % SB) * EXPERT_BYTES);
                // a copier thread stages + transfers + launches the streamed batches (it alone enqueues on cs
                // until joined) while this thread computes the CPU experts
                std::thread copier([&] {
                    for (int b0 = 0, bi = 0; b0 < (int)streamed.size(); b0 += SB, bi++) {
                        const int half_ = bi & 1, nb = std::min(SB, (int)streamed.size() - b0);
                        uint8_t* base = stage_p + (size_t)half_ * SB * EXPERT_BYTES;
                        uint8_t* hbase = hstage + (size_t)half_ * SB * EXPERT_BYTES;
                        double q0 = now();
                        CK(cudaEventSynchronize(ev_hfree[half_]));  // the DMA out of this host half is done
                        double q1 = now();
                        std::function<void(int)> cp = [&](int i) {
                            int key = l * E + streamed[b0 + i];
                            memcpy(hbase + (size_t)i * EXPERT_BYTES, eq[key], EXPERT_Q_BYTES);
                            memcpy(hbase + (size_t)i * EXPERT_BYTES + EXPERT_Q_BYTES, es[key], EXPERT_BYTES - EXPERT_Q_BYTES);
                        };
                        cpool.parallel_for(nb, cp);
                        pt[3] += q1 - q0; pt[4] += now() - q1;
                        CK(cudaStreamWaitEvent(cpp, ev_free[half_]));  // the previous batch in this VRAM half is computed
                        CK(cudaMemcpyAsync(base, hbase, (size_t)nb * EXPERT_BYTES, cudaMemcpyHostToDevice, cpp));
                        CK(cudaEventRecord(ev_hfree[half_], cpp));
                        CK(cudaEventRecord(ev_copied[half_], cpp));
                        CK(cudaStreamWaitEvent(cs, ev_copied[half_]));
                        pairs_forward(items_d + nres + b0, nb, pairs_d, hc, actp, ypair, cs);
                        CK(cudaEventRecord(ev_free[half_], cs));
                    }
                });
                streamed_n += streamed.size();
                // CPU: non-resident experts, each read once for all of its tokens, overlapped with the above
                double c1 = now();
                cpu.run_pairs(jobs, rows_h, T, T * TOPK, ypair_cpu.data());
                for (int t = 0; t < T; t++) {
                    float* y = ycpu_h + (size_t)t * D;
                    std::fill(y, y + D, 0.f);
                    for (int r = 0; r < TOPK; r++) {
                        int e = ridxc_h[t * TOPK + r];
                        if (job_of[e] < 0) continue;
                        const float* yp = ypair_cpu.data() + (size_t)(t * TOPK + r) * D;
                        for (int d = 0; d < D; d++) y[d] += yp[d];
                    }
                }
                cpu_s += now() - c1;
                pt[2] += now() - c1;
                double j0 = now();
                copier.join();
                pt[1] += now() - j0;  // time the main thread waited on the streaming
                pairs_reduce_k<<<dim3((D + 255) / 256, T), 256, 0, cs>>>(ypair, ycpu_d, moec, TOPK, D);
                gated_add_rows_k<<<dim3((D + 255) / 256, T), 256, 0, cs>>>(moec, shs, shl, D);
            }
            if (c0 + T == n) {  // the last prompt token: its logits pick the first generated token
                rmsnorm(Xc + (size_t)(T - 1) * D, moec + (size_t)(T - 1) * D, norm, h, D, EPS, 1, cs);
                gemv(lm_head, h, logits, D, V, cs);
                argmax(logits, V, amv, ams, ids_hd, cs);
                CK(cudaStreamSynchronize(cs));
                next = ids_h[0];
            }
        }
        fprintf(stderr, "[prefill] %d tok in %.2f s = %.1f tok/s | host: gpu-wait %.2f  streaming-wait %.2f  cpu-experts %.2f s | streamed %lld experts"
                " | copier: wait-dma %.2f  memcpy %.2f s | page faults %lu\n",
                n, now() - pt0, n / (now() - pt0), pt[0], pt[1], pt[2], streamed_n - sn0, pt[3], pt[4], page_faults() - pf0);
        return next;
    }

    int step(int tok, int pos, bool want_logits) {
        tokens++;
        double t = now();
        for (int i = 0; i < D; i++) h_h[i] = __half2float(embed[(size_t)tok * D + i]);
        copy_k<<<(D + 255) / 256, 256, 0, cs>>>(x, h_hd, D);  // zero-copy from the mapped host row
        set_int_k<<<1, 1, 0, cs>>>(pos_d, pos);
        for (int l = 0; l < NL; l++) {
            double h0 = now();
            poll_admissions();
            flush_table();  // the GPU's view of residency for this layer == the host's ready[] below
            if (prof || graphs.empty()) enqueue_layer(l); else CK(cudaGraphLaunch(graphs[l], cs));
            double h1 = now();
            CK(cudaEventSynchronize(ev_route));  // host needs the routing only for the misses
            double h2 = now();
            ht[0] += h1 - h0; ht[1] += h2 - h1;
            mark(0, t);
            if (trace) {
                int16_t rec[12] = {(int16_t)trace_prompt, (int16_t)pos, (int16_t)l, (int16_t)trace_phase};
                for (int j = 0; j < TOPK; j++) rec[4 + j] = (int16_t)ids_h[j];
                fwrite(rec, sizeof rec, 1, trace);
            }
            clock++;
            int nm = 0, mkey[TOPK];
            const uint8_t* mq[TOPK]; const float* msc[TOPK]; float mw[TOPK];
            for (int j = 0; j < TOPK; j++) {
                int key = l * E + ids_h[j], sl = slot_of[key];
                touch(key);
                if (sl >= 0) last[sl] = clock;  // protects this step's experts (incl. in-flight) from eviction
                if (sl >= 0 && ready[sl]) hits++;
                else { mq[nm] = eq[key]; msc[nm] = es[key]; mw[nm] = w_h[j]; mkey[nm++] = sl < 0 ? key : -1; misses++; }
            }
            // CPU: misses, overlapped with the GPU's experts + shared expert
            if (nm) {
                double c0 = now();
                cpu.run(nm, mq, msc, mw, h_h, y_h);
                double c1 = now();
                cpu_s += c1 - c0; ht[2] += c1 - c0;
                add_k<<<(D + 255) / 256, 256, 0, cs>>>(moe, y_hd, D);  // reads the host buffer over PCIe
                for (int j = 0; j < nm; j++) if (mkey[j] >= 0) admit(mkey[j]);
                start_admissions();
                ht[3] += now() - c1;
            }
            mark(1, t);
        }
        const float* pending = moe;
        // the last MoE output is folded into x here, or by the next token's first norm
        rmsnorm(x, pending, norm, h, D, EPS, 1, cs);
        if (!want_logits) return -1;
        gemv(lm_head, h, logits, D, V, cs);
        argmax(logits, V, amv, ams, ids_hd, cs);  // the id lands in mapped host memory
        CK(cudaStreamSynchronize(cs));
        mark(2, t);
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
    bool prof = false, f16 = false, use_prefill = true;
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
        else if (s == "--no-prefill") use_prefill = false;
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
        const int np = (int)ids.size();
        if (use_prefill) pos = np - 1;  // the prompt runs as one prefill; compare from its last position on
        for (;; pos++) {
            auto ref = read_f32(check + "/p" + std::string(pos < 10 ? "00" : pos < 100 ? "0" : "") + std::to_string(pos) + "_logits.bin");
            if (ref.empty()) break;
            int id = use_prefill && pos == np - 1 ? eng.prefill(ids.data(), np, 0) : eng.step(tok, pos, true);
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
            tok = pos + 1 < np ? ids[pos + 1] : id;
        }
        // F16 must be numerically identical; int8 is judged by distribution drift (KL) and greedy agreement
        // (prefill rounds activations to fp16 on tensor cores, so only the token-by-token F16 path is held to exactness)
        bool ok = pos > 0 && bad == 0 && (f16 && !use_prefill ? worst < 1e-4 : f16 ? worst_kl < 1e-5 : worst_kl < 0.02);
        fprintf(stderr, "[check] %s, %d positions, worst logits rel %.3e, worst KL %.2e, argmax mismatches %d -> %s\n",
                f16 ? "F16" : "int8", pos, worst, worst_kl, bad, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    struct Stats { int dec = 0; double sec = 0; long long hits = 0, misses = 0, admits = 0; double cpu = 0, ht[4] = {}, ttft = 0; };
    // Prompt token-by-token, then up to n greedy tokens. Returns decode-phase stats.
    auto generate = [&](const std::vector<int>& p, std::vector<int>& out, bool stream) {
        Stats st;
        if ((int)p.size() + n > ctx) { fprintf(stderr, "prompt + n exceeds --ctx %d\n", ctx); exit(2); }
        eng.trace_phase = 0;
        double t0 = now();
        int pos = 0, tok = -1;
        if (use_prefill && !eng.trace) { tok = eng.prefill(p.data(), (int)p.size(), 0); pos = (int)p.size(); }
        else for (; pos < (int)p.size(); pos++) tok = eng.step(p[pos], pos, pos + 1 == (int)p.size());
        long long h0 = eng.hits, m0 = eng.misses, a0 = eng.admits; double c0 = eng.cpu_s, t1 = now();
        double ht0[4]; for (int i = 0; i < 4; i++) ht0[i] = eng.ht[i];
        eng.trace_phase = 1;
        for (int gen = 0; gen < n; gen++) {
            out.push_back(tok);
            if (stream) { printf("%d\n", tok); fflush(stdout); }
            if (std::find(stop.begin(), stop.end(), tok) != stop.end() || gen + 1 == n) break;
            tok = eng.step(tok, pos++, true);
            st.dec++;
        }
        st.ttft = t1 - t0;
        st.sec = now() - t1; st.hits = eng.hits - h0; st.misses = eng.misses - m0; st.admits = eng.admits - a0; st.cpu = eng.cpu_s - c0;
        for (int i = 0; i < 4; i++) st.ht[i] = eng.ht[i] - ht0[i];
        return st;
    };
    auto report = [&](const char* tag, const Stats& st) {
        int dec = std::max(st.dec, 1);
        fprintf(stderr, "[%s] %d tok in %.2f s = %.2f tok/s | expert hit %.1f%% | %.1f misses/tok on CPU = %.1f ms/tok | %lld admitted\n",
                tag, st.dec, st.sec, st.dec / std::max(st.sec, 1e-9), 100.0 * st.hits / std::max(st.hits + st.misses, 1LL),
                (double)st.misses / dec, 1e3 * st.cpu / dec, st.admits);
        fprintf(stderr, "    host ms/tok: launch %.2f  wait-routing %.2f  cpu-misses %.2f  admissions %.2f | TTFT %.2f s\n",
                1e3 * st.ht[0] / dec, 1e3 * st.ht[1] / dec, 1e3 * st.ht[2] / dec, 1e3 * st.ht[3] / dec, st.ttft);
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
            for (int k = 0; k < 4; k++) tot.ht[k] += st.ht[k];
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
        const char* nm[] = {"layers (gpu)", "misses (cpu)", "lm_head+argmax"};
        double tot = 0; for (int i = 0; i < 3; i++) tot += eng.tp[i];
        for (int i = 0; i < 3; i++) fprintf(stderr, "  %-15s %7.2f ms/tok  %5.1f%%\n", nm[i], 1e3 * eng.tp[i] / std::max(st.dec, 1), 100 * eng.tp[i] / tot);
    }
    if (eng.trace) fclose(eng.trace);
    return 0;
}
