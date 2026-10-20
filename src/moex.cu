// MoEx v2 decode engine. Per MoE layer: resident experts run on the GPU, misses run on the CPU
// at the same time, and missed experts are copied into VRAM slots in the background (admission).
//   moex.exe --ids 760,6511,314 [--n 32] [--ctx 2048] [--stop 248044,248046] [--profile]
//   moex.exe --check ref/golden          logits vs the PyTorch reference at every golden position
//   --f16                                 keep dense weights F16 (exact); default is int8 per-row
//   --temp T --top-k K --top-p P --repeat-penalty R --seed S   sampling (default: greedy)
//   moex.exe --batch ids.txt [--trace routes.bin]   one prompt (comma ids) per line; expert cache stays warm
//     trace record: int16[12] = prompt, pos, layer, phase (0 prompt / 1 decode), 8 routed expert ids
//   --trace-ahead ahead.bin   per trace record: int16[5][16] = top-16 experts of layer l+k's router on layer l's
//     hidden state, k = 0..4 (-1 past the last layer); k = 0 reproduces the routing (a self-check)
// Generated token ids go to stdout one per line; stats go to stderr. Tokenizing is moex.py's job.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <cstdio>
#include <deque>
#include <list>
#include <memory>
#include <random>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "st.h"
#include "kernels.cuh"
#include "cpu_expert.h"
#include "spec.h"
#include "dense_ovr.h"
#include <nvtx3/nvToolsExt.h>  // header-only; ranges show in Nsight Systems, no-ops otherwise
#ifdef _WIN32
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
static unsigned long page_faults() { PROCESS_MEMORY_COUNTERS c{}; GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof c); return c.PageFaultCount; }
#else
static unsigned long page_faults() { rusage u{}; getrusage(RUSAGE_SELF, &u); return u.ru_minflt + u.ru_majflt; }
#endif

__global__ void copy_ids_k(const int* __restrict__ s, int* __restrict__ d) { d[threadIdx.x] = s[threadIdx.x]; }

// memcpy with non-temporal stores: no read-for-ownership of the destination, so the copy into the pinned ring takes
// 2 units of RAM traffic instead of 3 and leaves more bandwidth to the PCIe DMA reading it. d: 32-byte aligned, n % 32 == 0.
static void copy_nt(uint8_t* d, const void* s, size_t n) {
    const __m256i* sv = (const __m256i*)s;
    for (size_t i = 0; i < n / 32; i++) _mm256_stream_si256((__m256i*)d + i, _mm256_loadu_si256(sv + i));
    _mm_sfence();
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Qwen3.6-35B-A3B geometry (checked against the container shapes at load).
constexpr int D = 2048, NL = 40, E = 256, TOPK = 8, SI = 512, V = 248320;
constexpr int AH = 16, AKV = 2, ROT = 64, ASPLIT = 32;  // attention; ASPLIT: decode attention splits per kv head (test_kernels K7G)
constexpr int DC = 8192, DVH = 32, DKH = 16, CONVK = 4;  // DeltaNet
constexpr float EPS = 1e-6f, THETA = 1e7f;
// Top-8 router weights renormalized to sum to 1: the HF config omits norm_topk_prob, the Qwen3.5-MoE default is
// true, and llama.cpp normalizes for this architecture (colibri's converter wrote false into its meta).
static bool NORM_TOPK = !getenv("MOEX_NO_NORM_TOPK");
static bool PF_GEMV = getenv("MOEX_PF_GEMV") != nullptr;  // prefill experts on the old GEMV kernels (A/B tests)
static bool is_attn(int l) { return l % 4 == 3 || l == NL; }  // layer NL = the MTP head's layer
constexpr int NKEY = (NL + 1) * E;  // expert keys: 40 layers + the MTP layer
static const char* VERSION = "0.1.4";
// Too little RAM to lock every expert (the OS and other programs need ~5 GB): experts come from the SSD as needed.
static bool low_ram_machine() {
    MEMORYSTATUSEX ms{sizeof ms};
    GlobalMemoryStatusEx(&ms);
    return ms.ullTotalPhys < (size_t)NL * E * EXPERT_BYTES + (5ull << 30) || getenv("MOEX_LOW_RAM");
}

struct LayerW {
    half *in_ln, *post_ln, *router, *sh_gate, *qn, *kn, *conv, *nw, *alog, *dtb;  // small: F16 always
    // big: int8 unless --f16. Projections that share an input are one fused matrix (one launch, less tail):
    Mat sh_gu, sh_d, attn_qkv, o, dn_in, out;  // sh_gu=[gate;up], attn_qkv=[q;k;v], dn_in=[qkv;z;a;b]
    KVCache kv;
    float *ring, *S;                                          // state
};

struct Engine {
    Container C;
    std::vector<LayerW> L;
    Mat lm_head;
    half* norm;
    bool f16 = false;
    bool pin_async = false;  // chat/serve: lock the experts in RAM on a background thread (fast startup)
    std::thread pinner;
    half* qtmp = nullptr;  // F16 staging for quantization, freed after load
    const half* embed;  // mmapped, stays in RAM
    int ctx;
    // MTP (multi-token prediction): the model's own draft head (mtp.safetensors from tools/fetch_mtp.py). Each decode
    // round runs [token, draft] through the model as one 2-token chunk; a right draft gives 2 tokens for ~1.3x the cost.
    bool has_mtp = false, want_mtp = false;
    Mat mtp_fc;
    half *mtp_ne = nullptr, *mtp_nh = nullptr, *mtp_norm = nullptr;
    float *mtp_cat = nullptr, *lg2 = nullptr, *snap = nullptr, *dlog = nullptr;
    uint8_t* vbuf = nullptr;  // the 2-token round's scratch (the prefill scratch lives in expert slots)
    int draft = -1;  // the MTP head's draft
    // MOEX_NGRAM=n (opt-in, n = 2..4): prompt-lookup drafts (src/spec.h). When the context's last n..4 tokens occurred
    // earlier in it, the round drafts up to MOEX_NGRAM_MAX (default 4, max VT - 1) continuation tokens instead of the
    // head's one; the length adapts to the recent acceptance (DraftLen). The head still runs every round (it keeps
    // its own KV rows filled) and drafts when there is no match. Greedy drafts the most frequent continuation;
    // sampling draws from the continuation counts. Every draft is verified with p/q speculative sampling.
    NgramDrafter ngram{getenv("MOEX_NGRAM") ? std::max(0, atoi(getenv("MOEX_NGRAM"))) : 0};
    DraftLen dlen{getenv("MOEX_NGRAM_MAX") ? std::max(1, std::min(VT - 1, atoi(getenv("MOEX_NGRAM_MAX")))) : 4};
    std::vector<int> drafts;                // this round's drafts
    std::vector<std::vector<TokP>> dqs;     // the distribution each was drawn from (one-hot: head, greedy, --check)
    long long ng_rounds = 0, ng_acc = 0, ng_drafted = 0, ng_gated = 0;
    bool ng_draft = false;
    // Per prompt (reset by prefill): n-gram drafts only while their tokens per ms of round beat the head's (DraftGate;
    // deterministic mode prices a round by its rows instead of its wall time, so the choice is repeatable)
    DraftGate gate;
    void choose_draft(int head, int last) {
        draft = head;
        if (ngram.nmin) ngram.chain(last, dlen.k(), samp.temp <= 0.f, rng, drafts, dqs); else drafts.clear();
        if (!drafts.empty() && !gate.use_ngram()) { drafts.clear(); ng_gated++; }
        if (!(ng_draft = !drafts.empty())) { drafts = {head}; dqs = {{{head, 1.0}}}; }
    }
    void set_drafts(const std::vector<int>& d) {  // --check: teacher-forced drafts
        drafts = d; dqs.clear(); ng_draft = false;
        for (int t : d) dqs.push_back({{t, 1.0}});
    }
    // scratch
    float *rlog, *x, *h, *tmp, *moe, *big, *q, *k, *v, *qo, *ag, *apart, *z, *a, *b, *conv, *o, *act, *scr, *logits, *rw, *amv;
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
    // the same per layer, split by whether the layer had a CPU miss: [0] all experts in VRAM, [1] >= 1 miss; [.][4] = layers
    double hs[2][5] = {};
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
    unsigned *hmask_h, *hmask_hd;  // the GPU's hits among this layer's top-K (route_hits_k), so host and GPU always agree
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
    static constexpr int NSTAGE = 96;  // the step graph admits a whole token's misses at once (~75 at 1K)
    int nstage = NSTAGE;               // stages in use (load())
    struct Stage {
        uint8_t* buf; cudaEvent_t after_compute, done, back_done; int slot = -1, key = -1; int back_io = -1; uint8_t *back_q, *back_s;
        std::atomic<int> back_ready{0};  // the backer thread has enqueued this victim's copy-back (back_done recorded)
    };
    Stage stage[NSTAGE];
    std::vector<int> free_st;                 // main thread only
    std::deque<int> todo, flying, backs;      // guarded by mu
    std::mutex mu;
    std::thread loader, backer;
    cudaStream_t cb = nullptr;  // VRAM victim -> RAM copy-backs (own RAM cache, exclusive)
    bool stop_loader = false;
    FILE* trace = nullptr;
    FILE* trace_ahead = nullptr;
    float *ahead_h = nullptr, *ahead_d = nullptr;  // [D] normed hidden state, then [5][E] router scores
    int trace_prompt = 0, trace_phase = 0;
    // profile (synchronizing, so it perturbs the speed it measures)
    bool prof = false;
    double tp[8] = {};

    explicit Engine(const char* root, int ctx_) : C(root), ctx(ctx_) {}
    ~Engine() {
        if (pinner.joinable()) pinner.join();
        { std::lock_guard<std::mutex> g(mu); stop_loader = true; }
        if (loader.joinable()) loader.join();
        if (backer.joinable()) backer.join();
    }

    void loader_main() {
        const uint8_t* lq = nullptr; const float* ls = nullptr;
        while (true) {
            int i = -1;
            {
                std::lock_guard<std::mutex> g(mu);
                if (stop_loader) return;
                if (!todo.empty()) { i = todo.front(); todo.pop_front(); }
                if (i >= 0) { rc_loader_key = stage[i].key; lq = eq[stage[i].key]; ls = es[stage[i].key]; }
            }
            if (i < 0) { std::this_thread::sleep_for(std::chrono::microseconds(50)); continue; }
            Stage& st = stage[i];
            prefetch_experts(&st.key, 1);
            memcpy(st.buf, lq, EXPERT_Q_BYTES);  // may page-fault from SSD: this thread absorbs it
            memcpy(st.buf + EXPERT_Q_BYTES, ls, EXPERT_BYTES - EXPERT_Q_BYTES);
            { std::lock_guard<std::mutex> g(mu); rc_loader_key = -1; }
            if (st.back_io >= 0) {  // the victim's copy-back (backer thread, on cb) must finish before the overwrite
                while (!st.back_ready.load(std::memory_order_acquire)) std::this_thread::yield();
                CK(cudaStreamWaitEvent(cp, st.back_done));
            }
            CK(cudaStreamWaitEvent(cp, st.after_compute));  // nothing still reads the victim slot
            CK(cudaMemcpyAsync(arena + (size_t)st.slot * EXPERT_BYTES, st.buf, EXPERT_BYTES, cudaMemcpyHostToDevice, cp));
            if (!deterministic) {  // the GPU may use it from the next layer on (the host follows the GPU's hit mask)
                uint8_t** pub = (uint8_t**)(st.buf + EXPERT_BYTES);  // pinned, untouched until st.done
                *pub = arena + (size_t)st.slot * EXPERT_BYTES;
                CK(cudaMemcpyAsync(table_d + st.key, pub, sizeof(void*), cudaMemcpyHostToDevice, cp));
            }
            CK(cudaEventRecord(st.done, cp));
            std::lock_guard<std::mutex> g(mu);
            flying.push_back(i);
        }
    }

    // Copy-backs get their own thread and stream: enqueued by the loader they queued behind its admissions (each may
    // wait on an SSD page fault), and the CPU often needs the evicted expert right away (PLAN.md, 2.18 ms/token).
    // The slot holds a finished expert (ready), so the copy needs no wait; the admission's H2D waits for back_done.
    void backer_main() {
        while (true) {
            int i = -1;
            {
                std::lock_guard<std::mutex> g(mu);
                if (!backs.empty()) { i = backs.front(); backs.pop_front(); }
                else if (stop_loader) return;  // drained first: the loader may be waiting on one of these
            }
            if (i < 0) { std::this_thread::sleep_for(std::chrono::microseconds(20)); continue; }
            Stage& st = stage[i];
            const uint8_t* src = arena + (size_t)st.slot * EXPERT_BYTES;
            CK(cudaMemcpyAsync(st.back_q, src, EXPERT_Q_BYTES, cudaMemcpyDeviceToHost, cb));
            CK(cudaMemcpyAsync(st.back_s, src + EXPERT_Q_BYTES, EXPERT_BYTES - EXPERT_Q_BYTES, cudaMemcpyDeviceToHost, cb));
            CK(cudaEventRecord(rc_io[st.back_io].ev, cb));
            CK(cudaEventRecord(st.back_done, cb));
            rc_issued[st.back_io].store(1, std::memory_order_release);
            st.back_ready.store(1, std::memory_order_release);
        }
    }

    // Low-RAM mode (all experts don't fit in RAM, e.g. 16 GB machines): experts stay memory-mapped and Windows'
    // file cache keeps the recently used ones. Replaying route traces (tools/ram_sim.py), that recency policy is
    // close to the best possible (8 GB of experts: 20 SSD reads/token; locking a fixed hot set: 41), so what matters
    // is how each read happens: page faults pull an expert in small pieces (~2.4 ms, K14) while one large read of
    // 1.77 MB takes ~0.3-0.5 ms on NVMe. So every layer's missing experts are requested together, as large
    // parallel reads, as soon as the routing is known.
    bool low_ram = false;
    void prefetch_experts(const int* keys, int n) {
        if (!low_ram || n <= 0) return;
        std::vector<WIN32_MEMORY_RANGE_ENTRY> r;
        for (int i = 0; i < n; i++) {
            r.push_back({(void*)eq[keys[i]], EXPERT_Q_BYTES});
            r.push_back({(void*)es[keys[i]], EXPERT_BYTES - EXPERT_Q_BYTES});
        }
        PrefetchVirtualMemory(GetCurrentProcess(), r.size(), r.data(), 0);  // a hint: failure only costs speed
    }
    // ---- Own RAM expert cache (low-RAM mode, on by default; PLAN.md "16 GB machines: router-ahead prefetch") ----
    // Pinned slots filled by direct (unbuffered, overlapped) SSD reads instead of Windows' file cache, LRU. Size: the
    // RAM free at load minus 2 GB, or MOEX_RAM_GB=N (0 = off: Windows' file cache). VRAM and RAM hold different experts
    // (an expert admitted to VRAM leaves RAM, one evicted from VRAM is copied back over PCIe; MOEX_VICTIM=0 = off).
    // Layer l's graph predicts layer l+k's top-8 experts by its router on layer l's hidden state; the host prefetches
    // them at layer l+1 (MOEX_AHEAD=k, default 3: two layers before they're needed; 0 = off).
    // An expert not in the cache is still read through the file map (slow, correct).
    // The file offsets aren't 4 KB aligned: each part is read as its aligned superset, so a slot is [Q+4K][S+4K].
    static constexpr size_t RQ = EXPERT_Q_BYTES + 4096, RS = EXPERT_BYTES - EXPERT_Q_BYTES + 4096, RSLOT = RQ + RS;
    int rc_n = 0, ahead_k = 0;
    bool rc_victim = false;
    uint8_t* rc_base = nullptr;
    std::vector<const uint8_t*> mapq; std::vector<const float*> maps;  // file-map pointers: the fallback
    std::vector<int> rc_slot, rc_key, rc_io_of;  // key -> slot; slot -> key; slot -> io record in flight (-1)
    std::vector<char> rc_pf;                     // slot filled by a prefetch, not used yet
    std::list<int> rc_lru; std::vector<std::list<int>::iterator> rc_pos;  // slots in use, front = least recent
    std::vector<int> rc_free, rc_drops;
    std::vector<HANDLE> rc_files; std::vector<int> rc_fi; std::vector<uint64_t> rc_oq, rc_os;
    struct RcIo { OVERLAPPED ov[2]; cudaEvent_t ev; int slot = -1; bool pf = false, d2h = false; };
    std::vector<RcIo> rc_io; std::vector<int> rc_io_free, rc_flying;
    std::unique_ptr<std::atomic<int>[]> rc_issued{new std::atomic<int>[64]};  // copy-back enqueued by the loader
    int rc_loader_key = -1, rc_pf_flying = 0;
    long long rc_demand = 0, rc_pf_n = 0, rc_pf_used = 0, rc_backs = 0;
    double rc_wait = 0, rc_submit = 0;  // waiting for reads; main-thread time inside ReadFile
    double rc_wk[3] = {};               // of the wait: for a copy-back, a prefetch still in flight, a demand read
    int* pidx_h = nullptr; int *pidx_hd = nullptr, *pa_idx = nullptr; float *pa_h = nullptr, *pa_s = nullptr, *pa_w = nullptr;

    uint8_t* rcq(int s) { return rc_base + (size_t)s * RSLOT; }
    uint8_t* rcs(int s) { return rc_base + (size_t)s * RSLOT + RQ; }
    void rc_setup() {
        const char* gb = getenv("MOEX_RAM_GB");
        if (!low_ram) return;
        if (has_mtp) { if (gb) fprintf(stderr, "[ram] MOEX_RAM_GB is not supported with MTP: ignored\n"); return; }
        // the dense weights went to VRAM through the file map: hand their pages back first, so they count as free
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
        MEMORYSTATUSEX ms{sizeof ms};
        GlobalMemoryStatusEx(&ms);
        const double want = gb ? atof(gb) : ms.ullAvailPhys / 1e9 - 2.0;  // 2 GB: the OS and other programs
        if (want < 1.0) return;  // too little to beat the file cache
        rc_n = std::min((int)(want * 1e9 / RSLOT), NL * E);
        rc_base = (uint8_t*)VirtualAlloc(nullptr, (size_t)rc_n * RSLOT, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!rc_base) { fprintf(stderr, "[ram] cannot allocate %.1f GB\n", want); exit(1); }
        for (size_t o = 0, step = (size_t)512 * RSLOT; o < (size_t)rc_n * RSLOT; o += step)  // pinned: async copy-back
            CK(cudaHostRegister(rc_base + o, std::min(step, (size_t)rc_n * RSLOT - o), cudaHostRegisterDefault));
        rc_victim = !getenv("MOEX_VICTIM") || atoi(getenv("MOEX_VICTIM"));
        ahead_k = getenv("MOEX_AHEAD") ? atoi(getenv("MOEX_AHEAD")) : 3;
        mapq = eq; maps = es;
        rc_slot.assign(NKEY, -1); rc_key.assign(rc_n, -1); rc_io_of.assign(rc_n, -1); rc_pf.assign(rc_n, 0); rc_pos.resize(rc_n);
        for (int i = rc_n; i--;) rc_free.push_back(i);
        std::unordered_map<std::string, int> fidx;
        rc_fi.assign(NKEY, -1); rc_oq.assign(NKEY, 0); rc_os.assign(NKEY, 0);
        for (int k = 0; k < NL * E; k++) {
            std::string ep = "model.layers." + std::to_string(k / E) + ".mlp.experts." + std::to_string(k % E) + ".";
            const StTensor &tq = C.at(ep + "merged_weight"), &ts = C.at(ep + "qs");
            if (tq.file != ts.file) { fprintf(stderr, "[ram] expert %d spans two files\n", k); exit(1); }
            auto it = fidx.find(tq.file);
            if (it == fidx.end()) {
                HANDLE f = CreateFileA(tq.file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
                if (f == INVALID_HANDLE_VALUE) { fprintf(stderr, "[ram] cannot open %s\n", tq.file.c_str()); exit(1); }
                it = fidx.emplace(tq.file, (int)rc_files.size()).first; rc_files.push_back(f);
            }
            rc_fi[k] = it->second; rc_oq[k] = tq.off; rc_os[k] = ts.off;
        }
        rc_io.resize(64);
        for (int i = 0; i < 64; i++) {
            for (auto& o : rc_io[i].ov) o.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
            CK(cudaEventCreateWithFlags(&rc_io[i].ev, cudaEventDisableTiming));
            rc_io_free.push_back(i);
        }
        if (ahead_k) {
            CK(cudaHostAlloc((void**)&pidx_h, 2 * TOPK * 4, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&pidx_hd, pidx_h, 0));
            pa_h = falloc(D); pa_s = falloc(E); pa_w = falloc(TOPK); CK(cudaMalloc(&pa_idx, TOPK * 4));
        }
        fprintf(stderr, "[ram] own expert cache: %d slots (%.1f GB), direct SSD reads%s%s\n", rc_n, (double)rc_n * RSLOT / 1e9,
                rc_victim ? ", VRAM/RAM exclusive (victim copy-back)" : "", ahead_k ? (", prefetch " + std::to_string(ahead_k) + " layers ahead").c_str() : "");
    }
    // A slot for a new expert: a free one, else the least recently used one that is complete and not being copied.
    int rc_take() {
        if (!rc_free.empty()) { int s = rc_free.back(); rc_free.pop_back(); return s; }
        std::lock_guard<std::mutex> g(mu);  // vs the loader reading eq/es
        for (auto it = rc_lru.begin(); it != rc_lru.end(); ++it) {
            const int s = *it, k = rc_key[s];
            if (rc_io_of[s] >= 0 || k == rc_loader_key) continue;
            eq[k] = mapq[k]; es[k] = maps[k]; rc_slot[k] = -1; rc_key[s] = -1; rc_pf[s] = 0;
            rc_lru.erase(it);
            return s;
        }
        return -1;
    }
    void rc_use(int s) { rc_lru.erase(rc_pos[s]); rc_lru.push_back(s); rc_pos[s] = std::prev(rc_lru.end()); }
    int rc_start(int key, int& s) {  // io record + slot for key; -1 if none is free
        if (rc_io_free.empty() || (s = rc_take()) < 0) return -1;
        int i = rc_io_free.back(); rc_io_free.pop_back();
        rc_key[s] = key; rc_slot[key] = s; rc_io_of[s] = i;
        rc_lru.push_back(s); rc_pos[s] = std::prev(rc_lru.end());
        rc_io[i].slot = s; rc_flying.push_back(i);
        return i;
    }
    bool rc_read(int key, bool pf) {
        int s, i = rc_start(key, s);
        if (i < 0) return false;
        RcIo& io = rc_io[i]; io.pf = pf; io.d2h = false;
        const uint64_t off[2] = {rc_oq[key], rc_os[key]}, len[2] = {EXPERT_Q_BYTES, EXPERT_BYTES - EXPERT_Q_BYTES};
        uint8_t* dst[2] = {rcq(s), rcs(s)};
        const double t0 = now();
        for (int p = 0; p < 2; p++) {
            const uint64_t a0 = off[p] & ~4095ull, n = ((off[p] + len[p] + 4095) & ~4095ull) - a0;
            OVERLAPPED& o = io.ov[p];
            ResetEvent(o.hEvent); o.Internal = o.InternalHigh = 0; o.Offset = (DWORD)a0; o.OffsetHigh = (DWORD)(a0 >> 32);
            if (!ReadFile(rc_files[rc_fi[key]], dst[p], (DWORD)n, nullptr, &o) && GetLastError() != ERROR_IO_PENDING) {
                fprintf(stderr, "[ram] direct read failed (%lu)\n", GetLastError()); exit(1);
            }
        }
        rc_submit += now() - t0;
        if (pf) { rc_pf_flying++; rc_pf_n++; }
        return true;
    }
    // VRAM victim -> RAM, on cp ahead of the admission that overwrites it: enqueued by the loader thread with that
    // admission (st), or right here (st = null: the prefill slot claim, which then waits for cp)
    void rc_copy_back(int key, int vslot, Stage* st) {
        if (rc_slot[key] >= 0) return;
        int s, i = rc_start(key, s);
        if (i < 0) return;
        RcIo& io = rc_io[i]; io.pf = false; io.d2h = true;
        uint8_t *dq = rcq(s) + (rc_oq[key] & 4095), *ds = rcs(s) + (rc_os[key] & 4095);
        if (st) { rc_issued[i].store(0); st->back_ready.store(0); st->back_io = i; st->back_q = dq; st->back_s = ds; }
        else {
            const uint8_t* src = arena + (size_t)vslot * EXPERT_BYTES;
            CK(cudaMemcpyAsync(dq, src, EXPERT_Q_BYTES, cudaMemcpyDeviceToHost, cp));
            CK(cudaMemcpyAsync(ds, src + EXPERT_Q_BYTES, EXPERT_BYTES - EXPERT_Q_BYTES, cudaMemcpyDeviceToHost, cp));
            CK(cudaEventRecord(io.ev, cp));
            rc_issued[i].store(1);
        }
        rc_backs++;
    }
    bool rc_done(int i, bool wait) {
        RcIo& io = rc_io[i];
        if (io.d2h) {
            if (!rc_issued[i].load(std::memory_order_acquire)) {  // the loader hasn't enqueued it yet
                if (!wait) return false;
                while (!rc_issued[i].load(std::memory_order_acquire)) std::this_thread::yield();
            }
            if (wait) CK(cudaEventSynchronize(io.ev));
            else if (cudaEventQuery(io.ev) != cudaSuccess) return false;
        } else {
            for (int p = 0; p < 2; p++) {
                if (!wait && !HasOverlappedIoCompleted(&io.ov[p])) return false;
                DWORD n;
                if (!GetOverlappedResult(rc_files[rc_fi[rc_key[io.slot]]], &io.ov[p], &n, TRUE)) {
                    fprintf(stderr, "[ram] direct read failed (%lu)\n", GetLastError()); exit(1);
                }
            }
        }
        const int s = io.slot, k = rc_key[s];
        {
            std::lock_guard<std::mutex> g(mu);
            eq[k] = rcq(s) + (rc_oq[k] & 4095); es[k] = (const float*)(rcs(s) + (rc_os[k] & 4095));
        }
        rc_io_of[s] = -1; rc_pf[s] = io.pf;
        if (io.pf) rc_pf_flying--;
        rc_io_free.push_back(i);
        rc_flying.erase(std::find(rc_flying.begin(), rc_flying.end(), i));
        return true;
    }
    void rc_poll() {
        for (size_t j = rc_flying.size(); j--;) if (j < rc_flying.size()) rc_done(rc_flying[j], false);
        for (int k : rc_drops) {  // exclusive: admitted to VRAM, so the RAM copy goes
            const int s = rc_slot[k];
            if (s < 0 || rc_io_of[s] >= 0 || slot_of[k] < 0 || !ready[slot_of[k]]) continue;
            std::lock_guard<std::mutex> g(mu);
            if (k == rc_loader_key) continue;
            eq[k] = mapq[k]; es[k] = maps[k]; rc_slot[k] = -1; rc_key[s] = -1; rc_pf[s] = 0;
            rc_lru.erase(rc_pos[s]); rc_free.push_back(s);
        }
        rc_drops.clear();
    }
    // A layer's CPU misses: read the ones not in RAM (all together), wait for those still in flight.
    void rc_need(const int* keys, int n) {
        double t0 = now();
        for (int j = 0; j < n; j++) {
            const int k = keys[j], s = rc_slot[k];
            if (s >= 0) { if (rc_pf[s]) { rc_pf_used++; rc_pf[s] = 0; } rc_use(s); continue; }
            while (!rc_read(k, false)) rc_done(rc_flying.front(), true);  // no free io record: finish the oldest
            rc_demand++;
        }
        for (int j = 0; j < n; j++) {
            const int s = rc_slot[keys[j]];
            if (s < 0 || rc_io_of[s] < 0) continue;
            const int i = rc_io_of[s], kind = rc_io[i].d2h ? 0 : rc_io[i].pf ? 1 : 2;  // copy-back, prefetch, demand
            const double w0 = now();
            rc_done(i, true);
            rc_wk[kind] += now() - w0;
        }
        rc_wait += now() - t0;
    }
    void rc_prefetch(int l) {  // the previous layer's prediction, for layer l - 1 + ahead_k
        const int src = l - 1, lt = src + ahead_k;
        if (!ahead_k || src < 0 || lt >= NL) return;
        for (int j = 0; j < TOPK && rc_pf_flying < 8; j++) {
            const int k = lt * E + pidx_h[(src & 1) * TOPK + j];
            if (slot_of[k] >= 0 || rc_slot[k] >= 0) continue;
            if (!rc_read(k, true)) break;
        }
    }
    void rc_fill() {  // load: the most-used experts (not in VRAM, when exclusive) fill the cache
        if (!rc_n) return;
        double t0 = now();
        std::vector<int> keys;
        for (int k = 0; k < NL * E; k++) if (uses[k] && !(rc_victim && slot_of[k] >= 0)) keys.push_back(k);
        std::stable_sort(keys.begin(), keys.end(), [&](int a, int b) { return uses[a] > uses[b]; });
        int n = 0;
        for (int k : keys) {
            if (n >= rc_n) break;
            while (!rc_read(k, false)) rc_done(rc_flying.front(), true);
            n++;
        }
        while (!rc_flying.empty()) rc_done(rc_flying.front(), true);
        rc_submit = 0;
        fprintf(stderr, "[ram] own cache filled with the %d most-used experts%s, %.1f s\n", n, rc_victim ? " not in VRAM" : "", now() - t0);
    }
    void rc_report(long long toks) {
        if (!rc_n) return;
        const double t = (double)std::max(toks, 1LL);
        fprintf(stderr, "[ram] per token: %.1f demand reads, %.1f prefetched (%.1f used), %.1f copied back from VRAM | waiting for reads %.2f ms (copy-backs %.2f, late prefetches %.2f, "
                        "demand %.2f), issuing reads %.2f ms\n",
                rc_demand / t, rc_pf_n / t, rc_pf_used / t, rc_backs / t, 1e3 * rc_wait / t, 1e3 * rc_wk[0] / t, 1e3 * rc_wk[1] / t,
                1e3 * rc_wk[2] / t, 1e3 * rc_submit / t);
    }

    // Fills Windows' file cache with the most-used experts that fit in free RAM (not locked: the cache then
    // follows what is actually used).
    void prewarm_file_cache() {
        MEMORYSTATUSEX ms{sizeof ms};
        GlobalMemoryStatusEx(&ms);
        const size_t margin = 3ull << 30;  // the OS, other programs, and this process's own buffers
        const int n = (int)std::min<size_t>(NL * E, ms.ullAvailPhys > margin ? (ms.ullAvailPhys - margin) / EXPERT_BYTES : 0);
        std::vector<int> keys(NL * E);
        for (int i = 0; i < NL * E; i++) keys[i] = i;
        const std::vector<uint32_t> u = uses;  // snapshot: runs on the pinner thread while decode updates uses
        std::stable_sort(keys.begin(), keys.end(), [&](int a, int b) { return u[a] > u[b]; });
        double t0 = now();
        for (int i = 0; i < n; i += 256) prefetch_experts(keys.data() + i, std::min(256, n - i));
        fprintf(stderr, "[ram] %d of %d experts (%.1f GB) read into the file cache, %.1f s\n", n, NL * E, n * (double)EXPERT_BYTES / 1e9, now() - t0);
    }

    // A CPU miss on a mapped page this process hasn't touched costs ~2.4 ms vs ~0.06 ms touched
    // (measured, K14). So: read every expert into RAM up front, fault each page into the working
    // set once, and ask Windows to keep the working set that large.
    void pin_experts_in_ram() {
        double t0 = now();
        std::vector<WIN32_MEMORY_RANGE_ENTRY> r;
        size_t total = 0;
        for (int k = 0; k < NKEY; k++) {
            if (!eq[k]) continue;  // no MTP head
            r.push_back({(void*)eq[k], EXPERT_Q_BYTES});
            r.push_back({(void*)es[k], EXPERT_BYTES - EXPERT_Q_BYTES});
            total += EXPERT_BYTES;
        }
        MEMORYSTATUSEX ms{sizeof ms};
        GlobalMemoryStatusEx(&ms);
        if (low_ram) {
            if (ms.ullTotalPhys < total + (5ull << 30))
                fprintf(stderr, "[ram] %.1f GB of RAM is below the ~%.0f GB needed to keep all %.1f GB of experts in memory: "
                                "the most-used experts are cached in RAM and the rest are read from the SSD when needed (slower decode)\n",
                        ms.ullTotalPhys / 1e9, (total + (5ull << 30)) / 1e9, total / 1e9);
            else fprintf(stderr, "[ram] low-RAM mode forced by MOEX_LOW_RAM (for testing): experts are read from the SSD when needed\n");
            // tools/prefetch_sim.py: ~+2.3 tok/s per extra GB free for experts in the 6-10 GB range
            fprintf(stderr, "[ram] TIP: %.1f GB of RAM is free right now. Close browser tabs and other apps before starting moex:\n"
                            "      every extra GB of free RAM is about +2 tokens/s.\n", ms.ullAvailPhys / 1e9);
            // the dense weights (~4 GB) were read through the map and already live in VRAM: hand their pages back.
            // (A working-set minimum to stop re-faults was measured: 21 -> 17 tok/s under memory pressure.)
            SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
            if (!rc_n) prewarm_file_cache();  // else the own RAM cache is filled after load
            return;
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

    // MOEX_DETERMINISTIC=1 (greedy gates): every admission started so far is finished and published at each poll, so
    // residency, and with it which experts run on the CPU (int8 activations, other rounding than the GPU), depends
    // only on the tokens, not on copy timing: the run-to-run splits in PLAN.md. The kernels and the CPU pool already
    // reduce in a fixed order (no float atomics; each output element is one work item's loop). Costs the overlap of
    // admission copies with compute. Not for the low-RAM mode (own RAM cache reads are timing-dependent too).
    const bool deterministic = getenv("MOEX_DETERMINISTIC") != nullptr;
    // Publish finished admissions (main thread).
    void poll_admissions() {
        if (deterministic) {
            while (true) {  // the loader has taken and enqueued every job: each started stage is on `flying`
                { std::lock_guard<std::mutex> g(mu); if (todo.empty() && free_st.size() + flying.size() == (size_t)nstage) break; }
                std::this_thread::yield();
            }
            CK(cudaStreamSynchronize(cp));
        }
        std::lock_guard<std::mutex> g(mu);
        for (size_t n = flying.size(); n--;) {
            int i = flying.front(); flying.pop_front();
            if (cudaEventQuery(stage[i].done) == cudaSuccess) {
                Stage& st = stage[i];
                if (owner[st.slot] == st.key) {
                    ready[st.slot] = 1;
                    if (deterministic) table_set(st.key, arena + (size_t)st.slot * EXPERT_BYTES);  // else published by the copy
                    if (rc_victim) rc_drops.push_back(st.key);
                }
                inflight[st.slot] = 0;
                free_st.push_back(i);
            } else flying.push_back(i);
        }
    }

    // Eviction score: decayed use count with a 10-token half-life (log domain). Measured on 80-prompt
    // traces (tools/sim.py): 82.0% hit vs 80.8% LRU; longer half-lives lose (tau 200: 69%).
    static constexpr double TAU = 10.0;
    std::vector<double> G = std::vector<double>(NKEY, -INFINITY);

    // Warm start (--load): use counts per expert, summed over past sessions in hot_file (moex_hot.bin in the
    // model folder; the release ships one from our test prompts for the first run). The hottest experts fill
    // the VRAM slots at load instead of arriving one miss at a time during the first replies.
    std::vector<uint32_t> uses = std::vector<uint32_t>(NKEY, 0);
    std::string hot_file;
    bool warm = false;
    bool quiet = false;  // chat/serve: no per-prefill stats line
    void load_hot(const std::vector<std::string>& paths) {
        for (auto& p : paths) {
            FILE* f = fopen(p.c_str(), "rb");
            if (!f) continue;
            std::vector<uint32_t> u(NL * E);
            bool ok = fread(u.data(), 4, u.size(), f) == u.size();
            fclose(f);
            if (ok) { std::copy(u.begin(), u.end(), uses.begin()); return; }
        }
    }
    void save_hot() {
        FILE* f = fopen(hot_file.c_str(), "wb");
        if (!f) return;  // read-only model folder: the counts just aren't kept
        fwrite(uses.data(), 4, uses.size(), f);
        fclose(f);
    }
    void warm_slots() {
        std::vector<int> keys(NL * E);
        for (int i = 0; i < NL * E; i++) keys[i] = i;
        std::sort(keys.begin(), keys.end(), [&](int a, int b) { return uses[a] > uses[b]; });
        const double mx = std::max(uses[keys[0]], 1u);
        std::vector<const uint8_t*> tab(NKEY, nullptr);
        int n = 0;
        for (; n < nslots - nres && uses[keys[n]]; n++) {  // the tail slots belong to prefill
            const int key = keys[n];
            uint8_t* dst = arena + (size_t)n * EXPERT_BYTES;
            memcpy(stage[0].buf, eq[key], EXPERT_Q_BYTES);
            memcpy(stage[0].buf + EXPERT_Q_BYTES, es[key], EXPERT_BYTES - EXPERT_Q_BYTES);
            CK(cudaMemcpy(dst, stage[0].buf, EXPERT_BYTES, cudaMemcpyHostToDevice));
            owner[n] = key; slot_of[key] = n; ready[n] = 1; tab[key] = dst;
            G[key] = log(uses[key] / mx);  // <= 0: ranked like recent uses, then decays as usual
        }
        CK(cudaMemcpy(table_d, tab.data(), tab.size() * sizeof(void*), cudaMemcpyHostToDevice));
        fprintf(stderr, "[load] %d most-used experts preloaded into VRAM\n", n);
    }
    long long tokens = 0;
    void touch(int key) {
        uses[key]++;
        double v = tokens / TAU, g = G[key];
        G[key] = g == -INFINITY ? v : std::max(g, v) + log1p(exp(-fabs(g - v)));
    }

    // Start copying expert `key` into the lowest-scoring slot not used this layer step; skipped when all stages are busy.
    // Admit only experts whose decayed use count (10-token half-life) is >= admit_min: an admission moves ~3x an
    // expert's bytes through RAM (mmap -> pinned -> PCIe) while a CPU miss reads it once, and the CPU experts are
    // RAM-bandwidth-bound. 2.5 measured best at 1K (raw 50.8 -> 56.0, MTP 58.7 -> 64.6; PLAN.md). Low-RAM mode
    // (own RAM cache, SSD reads) keeps 0 = admit every miss. MOEX_ADMIT_MIN overrides.
    double admit_min = 2.5;
    void admit(int key) {
        if (free_st.empty()) return;
        if (admit_min > 0 && exp(G[key] - tokens / TAU) < admit_min) return;
        int s = -1; double best = INFINITY;
        for (int i = 0; i < nslots; i++) {
            double sc = owner[i] < 0 ? -INFINITY : G[owner[i]];
            if (!inflight[i] && last[i] != clock && sc < best) { best = sc; s = i; }
        }
        if (s < 0) return;
        const int i = free_st.back();
        stage[i].back_io = -1;
        if (owner[s] >= 0 && ready[s] && rc_victim) rc_copy_back(owner[s], s, &stage[i]);
        if (owner[s] >= 0) { slot_of[owner[s]] = -1; if (ready[s]) table_set(owner[s], nullptr); }
        owner[s] = key; slot_of[key] = s; ready[s] = 0; inflight[s] = 1; last[s] = clock;
        free_st.pop_back();
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
        for (int i : new_jobs) { todo.push_back(i); if (stage[i].back_io >= 0) backs.push_back(i); }
        new_jobs.clear();
    }

    // a then b, contiguous (decode reads the router and the shared-expert gate in one GEMV)
    half* wup2(const std::string& a, int64_t na, const std::string& b, int64_t nb) {
        if ((int64_t)C.at(a).bytes != na * 2 || (int64_t)C.at(b).bytes != nb * 2) { fprintf(stderr, "shape mismatch %s\n", a.c_str()); exit(1); }
        half* d; CK(cudaMalloc(&d, (na + nb) * 2));
        CK(cudaMemcpy(d, C.map(a), na * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d + na, C.map(b), nb * 2, cudaMemcpyHostToDevice));
        return d;
    }
    half* wup(const std::string& n, int64_t numel) {
        const StTensor& s = C.at(n);
        if ((int64_t)s.bytes != numel * 2) { fprintf(stderr, "shape mismatch %s\n", n.c_str()); exit(1); }
        half* d; CK(cudaMalloc(&d, s.bytes));
        CK(cudaMemcpy(d, C.map(n), s.bytes, cudaMemcpyHostToDevice));
        return d;
    }
    // Dense matrix groups stored int6 gs64 instead of int8 (--q6 groups, "none" = all int8; same names as --q4).
    // Default: the mix within the KL budget on eval1 + code1 + agent1 (PLAN.md "Step 1: 6-bit mix").
    std::string q6groups = "dn_out,lm_head,sh_d";
    bool is_q6(const char* g) const {
        std::string l = "," + q6groups + ",";
        return !f16 && l.find("," + std::string(g) + ",") != std::string::npos;
    }
    // Dense matrix groups stored int4 gs64 instead of int8 (--q4 dn_in,dn_out,attn_qkv,attn_o,sh_gu,sh_d,lm_head).
    std::string q4groups;
    bool is_q4(const char* g) const {
        std::string l = "," + q4groups + ",";
        return !f16 && l.find("," + std::string(g) + ",") != std::string::npos;
    }
    // MOEX_DENSE=dir (research): matrices found there in tools/quant_dense.py's packed formats (src/dense_ovr.h: int4 /
    // int6, gs 32 / 64, symmetric or with zero points, optional AWQ input scales) replace the container's; their group is
    // kept as F16 holding the dequantized values, so any packed file screens without a kernel.
    std::unique_ptr<Container> dense_ovr = getenv("MOEX_DENSE") ? std::make_unique<Container>(getenv("MOEX_DENSE")) : nullptr;
    Mat dense_override(std::initializer_list<std::pair<std::string, int64_t>> parts, int64_t I) {
        int64_t O = 0;
        for (auto& [n, rows] : parts) O += rows;
        std::vector<half> h((size_t)(O * I));
        int64_t off = 0;
        for (auto& [n, rows] : parts) {
            half* dst = h.data() + off * I;
            dequant_packed(*dense_ovr, n, rows, I, [&](int64_t k, float w) { dst[k] = __float2half(w); });
            off += rows;
        }
        half* f; CK(cudaMalloc(&f, h.size() * 2));
        CK(cudaMemcpy(f, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
        return {f, nullptr, nullptr, nullptr};
    }
    // Row-concatenation of matrices with the same input width I, as F16, int8 + per-row scales, or int4 gs64.
    Mat wmat(std::initializer_list<std::pair<std::string, int64_t>> parts, int64_t I, const char* group) {
        if (dense_ovr && !f16 && dense_ovr->has(parts.begin()->first + ".s")) return dense_override(parts, I);
        // research: MOEX_FQ=group:bits:gs:asym,... keeps that group as F16 holding fake-quantized values
        int fq[3] = {0, 0, 0};
        if (const char* e = getenv("MOEX_FQ")) {
            std::string l = std::string(",") + e;
            size_t k = l.find("," + std::string(group) + ":");
            if (k != std::string::npos) sscanf(l.c_str() + k + 2 + strlen(group), "%d:%d:%d", &fq[0], &fq[1], &fq[2]);
        }
        if (fq[0] && !f16) {
            int64_t O = 0; for (auto& [n, rows] : parts) O += rows;
            half* f; CK(cudaMalloc(&f, O * I * 2));
            int64_t off = 0;
            for (auto& [n, rows] : parts) { CK(cudaMemcpy(f + off * I, C.map(n), rows * I * 2, cudaMemcpyHostToDevice)); off += rows; }
            fakequant_rows_k<<<(int)O, 64>>>(f, (int)I, fq[0], fq[1], fq[2]);
            CK(cudaGetLastError());
            return {f, nullptr, nullptr, nullptr};
        }
        const bool q4 = is_q4(group);
        if (is_q6(group) && !q4) {
            int64_t O = 0; for (auto& [n, rows] : parts) O += rows;
            uint8_t *lo, *hi; float* sc;
            CK(cudaMalloc(&lo, O * I / 2)); CK(cudaMalloc(&hi, O * I / 4)); CK(cudaMalloc(&sc, O * (I / 64) * 4));
            int64_t off = 0;
            for (auto& [n, rows] : parts) {  // the synchronous copy orders qtmp's reuse after the previous quantize
                CK(cudaMemcpy(qtmp, C.map(n), rows * I * 2, cudaMemcpyHostToDevice));
                quantize_rows_q6_k<<<(int)rows, 64>>>(qtmp, lo + off * I / 2, hi + off * I / 4, sc + off * (I / 64), (int)I);
                CK(cudaGetLastError());
                off += rows;
            }
            Mat m; m.s = sc; m.q6 = lo; m.q6h = hi;
            return m;
        }
        int64_t O = 0;
        for (auto& [n, rows] : parts) {
            if ((int64_t)C.at(n).bytes != rows * I * 2) { fprintf(stderr, "shape mismatch %s\n", n.c_str()); exit(1); }
            O += rows;
        }
        half* f = nullptr; int8_t* q = nullptr; float* sc = nullptr; uint8_t* p4 = nullptr;
        if (f16) CK(cudaMalloc(&f, O * I * 2));
        else if (q4) { CK(cudaMalloc(&p4, O * I / 2)); CK(cudaMalloc(&sc, O * (I / 64) * 4)); }
        else { CK(cudaMalloc(&q, O * I)); CK(cudaMalloc(&sc, O * 4)); }
        int64_t off = 0;
        for (auto& [n, rows] : parts) {
            if (f16) CK(cudaMemcpy(f + off * I, C.map(n), rows * I * 2, cudaMemcpyHostToDevice));
            else {  // the synchronous copy also orders qtmp's reuse after the previous quantize
                CK(cudaMemcpy(qtmp, C.map(n), rows * I * 2, cudaMemcpyHostToDevice));
                if (q4) quantize_rows_q4_k<<<(int)rows, 64>>>(qtmp, p4 + off * I / 2, sc + off * (I / 64), (int)I);
                else quantize_rows_i8_k<<<(int)rows, 256>>>(qtmp, q + off * I, sc + off, (int)I);
                CK(cudaGetLastError());
            }
            off += rows;
        }
        return {f, q, sc, p4};
    }
    static float* falloc(size_t n, bool zero = false) {
        float* d; CK(cudaMalloc(&d, n * 4)); if (zero) CK(cudaMemset(d, 0, n * 4)); return d;
    }

    void load() {
        double t0 = now();
        gemv_q6_init();
        size_t vf[5], vtot; CK(cudaMemGetInfo(&vf[0], &vtot));  // VRAM audit: free before load, after weights, after the
                                                                // F16 staging is freed, after the scratch, after the cache
        embed = (const half*)C.map("model.embed_tokens.weight");
        if (!f16) CK(cudaMalloc(&qtmp, (size_t)V * D * 2));  // largest matrix
        lm_head = wmat({{"lm_head.weight", V}}, D, "lm_head");
        norm = wup("model.norm.weight", D);
        // MTP: +7-10% on a 6 GB GPU with every expert in RAM (PLAN.md), so main() turns it on there by default; on
        // low-RAM machines the draft token's extra experts become extra SSD reads (and the own RAM cache needs it off)
        has_mtp = C.has("mtp.fc.weight") && (want_mtp || getenv("MOEX_MTP"));
        if (want_mtp && !has_mtp) fprintf(stderr, "[mtp] --mtp needs the MTP head (mtp.safetensors): run `moex download` once more\n");
        L.resize(NL + has_mtp);
        for (int i = 0; i < NL + has_mtp; i++) {
            std::string p = i < NL ? "model.layers." + std::to_string(i) + "." : "mtp.layers.0.", m = p + "mlp.";
            LayerW& w = L[i];
            w.in_ln = wup(p + "input_layernorm.weight", D);
            w.post_ln = wup(p + "post_attention_layernorm.weight", D);
            w.router = wup2(m + "gate.weight", (int64_t)E * D, m + "shared_expert_gate.weight", D);  // row E: the shared gate
            w.sh_gu = wmat({{m + "shared_expert.gate_proj.weight", SI}, {m + "shared_expert.up_proj.weight", SI}}, D, "sh_gu");
            w.sh_d = wmat({{m + "shared_expert.down_proj.weight", D}}, SI, "sh_d");
            w.sh_gate = w.router + (size_t)E * D;
            if (is_attn(i)) {
                std::string a = p + "self_attn.";
                w.attn_qkv = wmat({{a + "q_proj.weight", AH * 2 * AHD}, {a + "k_proj.weight", AKV * AHD}, {a + "v_proj.weight", AKV * AHD}}, D, "attn_qkv");
                w.o = wmat({{a + "o_proj.weight", D}}, AH * AHD, "attn_o");
                w.qn = wup(a + "q_norm.weight", AHD);
                w.kn = wup(a + "k_norm.weight", AHD);
                w.kv = kv_alloc(AKV, ctx);
            } else {
                std::string a = p + "linear_attn.";
                w.dn_in = wmat({{a + "in_proj_qkv.weight", DC}, {a + "in_proj_z.weight", DVH * DNV},
                                {a + "in_proj_a.weight", DVH}, {a + "in_proj_b.weight", DVH}}, D, "dn_in");
                w.out = wmat({{a + "out_proj.weight", D}}, DVH * DNV, "dn_out");
                w.conv = wup(a + "conv1d.weight", (int64_t)DC * CONVK);
                w.nw = wup(a + "norm.weight", DNV);
                w.alog = wup(a + "A_log", DVH);
                w.dtb = wup(a + "dt_bias", DVH);
                w.ring = falloc((size_t)DC * (CONVK - 1), true);
                w.S = falloc((size_t)DVH * DNK * DNV, true);
            }
        }
        if (has_mtp) {
            mtp_fc = wmat({{"mtp.fc.weight", D}}, 2 * D, "mtp_fc");
            mtp_ne = wup("mtp.pre_fc_norm_embedding.weight", D);
            mtp_nh = wup("mtp.pre_fc_norm_hidden.weight", D);
            mtp_norm = wup("mtp.norm.weight", D);
        }
        CK(cudaMemGetInfo(&vf[1], &vtot));
        if (qtmp) { CK(cudaDeviceSynchronize()); CK(cudaFree(qtmp)); qtmp = nullptr; }
        CK(cudaMemGetInfo(&vf[2], &vtot));
        CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking));
        x = falloc(D); h = falloc(D); tmp = falloc(D); moe = falloc(D);
        q = falloc(AH * 2 * AHD + 2 * AKV * AHD); k = q + AH * 2 * AHD; v = k + AKV * AHD; qo = falloc(AH * AHD); ag = falloc(AH * AHD);
        big = falloc(DC + DVH * DNV + 2 * DVH); z = big + DC; a = z + DVH * DNV; b = a + DVH;
        conv = falloc(DC); o = falloc(DVH * DNV);
        act = falloc(TOPK * EI); scr = falloc(2 * SI + D + 1); rlog = falloc(E + 1); logits = falloc(V); rw = falloc(TOPK);
        amv = falloc(ARGMAX_BLOCKS + 1);
        CK(cudaMalloc(&ridx, TOPK * 4)); CK(cudaMalloc(&ams, ARGMAX_BLOCKS * 4));
        CK(cudaMalloc(&slots_d, TOPK * sizeof(void*)));
        apart = falloc(VT * AH * ASPLIT * (AHD + 2));  // decode, or the MTP round's rows
        alloc_prefill();
        if (has_mtp) {
            CK(cudaMalloc(&vbuf, carve_prefill(nullptr, VT)));
            mtp_cat = falloc((size_t)CH * 2 * D); lg2 = falloc((size_t)VT * V); dlog = falloc(V);
            snap_rows = ngram.nmin ? dlen.kmax : 1;
            snap = falloc(NDN * DNS); rsnap = falloc((size_t)snap_rows * RSTRIDE);
            if (snap_rows > 1) {  // replay inputs for rows 1 .. kmax - 1 (~1 MB per draft row)
                rp_conv = falloc(NDN * (snap_rows - 1) * DC); rp_ab = falloc(NDN * (snap_rows - 1) * 2 * DVH);
                rp_out = falloc((size_t)(snap_rows - 1) * DVH * DNV);
            }
            snap_bytes = (NDN * DNS + (size_t)snap_rows * RSTRIDE + (snap_rows > 1 ? NDN * (snap_rows - 1) * (DC + 2 * DVH) + (size_t)(snap_rows - 1) * DVH * DNV : 0)) * 4;
        }
        eq.assign(NKEY, nullptr); es.assign(NKEY, nullptr);
        for (int l = 0; l < NL + has_mtp; l++) for (int e = 0; e < E; e++) {
            std::string ep = (l < NL ? "model.layers." + std::to_string(l) + "." : std::string("mtp.layers.0.")) + "mlp.experts." + std::to_string(e) + ".";
            eq[l * E + e] = C.map(ep + "merged_weight");
            es[l * E + e] = (const float*)C.map(ep + "qs");
        }
        low_ram = low_ram_machine();
        admit_min = getenv("MOEX_ADMIT_MIN") ? atof(getenv("MOEX_ADMIT_MIN")) : low_ram ? 0.0 : admit_min;
        rc_setup();
        if (pin_async) pinner = std::thread([this] { pin_experts_in_ram(); });  // until done, misses may page-fault
        else pin_experts_in_ram();
        // every byte of VRAM left (minus headroom for the driver) becomes expert slots
        size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
        const size_t reserve = (getenv("MOEX_RESERVE_MB") ? atoi(getenv("MOEX_RESERVE_MB")) : 320) * (1ull << 20);  // env: testing smaller caches
        nslots = (int)((fr - std::min(fr, reserve)) / EXPERT_BYTES);
        nslots = std::min(nslots, NKEY + (int)((carve_prefill(nullptr) + EXPERT_BYTES - 1) / EXPERT_BYTES));  // big GPUs: every expert fits
        nres = (int)((carve_prefill(nullptr) + EXPERT_BYTES - 1) / EXPERT_BYTES);
        if (nslots < 3 * nres) { nring = 2; nres = (int)((carve_prefill(nullptr) + EXPERT_BYTES - 1) / EXPERT_BYTES); }  // small GPUs
        if (nslots < nres + 2 * TOPK) { fprintf(stderr, "only %d expert slots fit; need >= %d\n", nslots, 2 * TOPK); exit(1); }
        vf[3] = fr;
        CK(cudaMalloc(&arena, (size_t)nslots * EXPERT_BYTES));
        CK(cudaMemGetInfo(&vf[4], &vtot));
        {
            const double MB = 1 << 20;
            const size_t kv = (size_t)(NL / 4) * (2 * (size_t)AKV * ctx * AHD + 2 * (size_t)AKV * ctx * 4);
            const size_t dn = (size_t)(NL - NL / 4) * ((size_t)DC * (CONVK - 1) + (size_t)DVH * DNK * DNV) * 4;
            const size_t lmh = f16 ? (size_t)V * D * 2 : is_q6("lm_head") ? (size_t)V * D * 3 / 4 + (size_t)V * (D / 64) * 4 : (size_t)V * D + (size_t)V * 4;
            const size_t router = (size_t)(NL + has_mtp) * E * D * 2;
            const size_t weights = vf[0] - vf[1] - (f16 ? 0 : (size_t)V * D * 2) - kv - dn;  // minus the F16 staging
            const size_t snapb = snap_bytes;
            fprintf(stderr, "[vram] total %.0f MB, in use before load %.0f (CUDA context, display, other apps) | lm_head %.0f | "
                            "dense + MTP head %.0f (incl. routers f16 %.0f) | KV cache %.0f (ctx %d) | DeltaNet state %.0f | "
                            "decode/MTP scratch %.0f (incl. MTP snapshot %.0f) | expert cache %d slots = %.0f MB (prefill borrows %d) | "
                            "free %.0f (reserve)\n",
                    vtot / MB, (vtot - vf[0]) / MB, lmh / MB, (weights - lmh) / MB, router / MB, kv / MB, ctx, dn / MB,
                    (vf[2] - vf[3]) / MB, snapb / MB, nslots, nslots * (double)EXPERT_BYTES / MB, nres, vf[4] / MB);
        }
        carve_prefill(arena + (size_t)(nslots - nres) * EXPERT_BYTES);
        slot_of.assign(NKEY, -1); owner.assign(nslots, -1); last.assign(nslots, 0);
        ready.assign(nslots, 0); inflight.assign(nslots, 0);
        auto mapped = [](auto** hp, auto** dp, size_t bytes) {
            CK(cudaHostAlloc((void**)hp, bytes, cudaHostAllocMapped));
            CK(cudaHostGetDevicePointer((void**)dp, *hp, 0));
        };
        mapped(&hmask_h, &hmask_hd, 4);
        mapped(&h_h, &h_hd, D * 4); mapped(&y_h, &y_hd, D * 4); mapped(&w_h, &w_hd, TOPK * 4); mapped(&ids_h, &ids_hd, TOPK * 4);
        CK(cudaMalloc(&hw_d, TOPK * 4)); CK(cudaMalloc(&nh_d, 4));
        CK(cudaMalloc(&table_d, NKEY * sizeof(void*))); CK(cudaMemset(table_d, 0, NKEY * sizeof(void*)));
        CK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
        CK(cudaMalloc(&pos_d, 4));
        // the step graph admits a whole token's misses at once: 96 stages; the MTP round admits per layer, and more
        // stages there only add copy churn (32: -4% MTP tok/s at 96). MOEX_NSTAGE overrides (A/B).
        nstage = getenv("MOEX_NSTAGE") ? std::clamp(atoi(getenv("MOEX_NSTAGE")), 8, NSTAGE) : has_mtp ? 32 : NSTAGE;
        for (int i = 0; i < nstage; i++) {
            CK(cudaMallocHost(&stage[i].buf, EXPERT_BYTES + 64));  // + the slot pointer the copy publishes
            CK(cudaEventCreateWithFlags(&stage[i].after_compute, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stage[i].done, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&stage[i].back_done, cudaEventDisableTiming));
            free_st.push_back(i);
        }
        if (warm) warm_slots();
        rc_fill();
        loader = std::thread([this] { loader_main(); });
        if (rc_victim) { CK(cudaStreamCreateWithFlags(&cb, cudaStreamNonBlocking)); backer = std::thread([this] { backer_main(); }); }
        capture_graphs();
        fprintf(stderr, "[load] %.1f s, dense %s, %d expert slots (%.2f GB, %.1f%% of %d experts), ctx %d%s%s\n",
                now() - t0, f16 ? "F16" : ("int8" + (q6groups != "none" ? ", int6: " + q6groups : "") + (q4groups.empty() ? "" : ", int4: " + q4groups)).c_str(), nslots, nslots * (double)EXPERT_BYTES / 1e9, 100.0 * nslots / (NL * E), NL * E, ctx,
                has_mtp ? ", MTP on" : "", deterministic ? ", deterministic" : "");
    }

    // New conversation: recurrent state back to zero (KV rows are overwritten position by position).
    void reset() {
        recent.clear(); ngram.reset();
        for (int l = 0; l < NL; l++) if (!is_attn(l)) {
            CK(cudaMemsetAsync(L[l].ring, 0, (size_t)DC * (CONVK - 1) * 4, cs));
            CK(cudaMemsetAsync(L[l].S, 0, (size_t)DVH * DNK * DNV * 4, cs));
        }
    }

    void mark(int i, double& t) { if (prof) { CK(cudaStreamSynchronize(cs)); double n = now(); tp[i] += n - t; t = n; } }

    // All of layer l's GPU work: mixer, post norm, router (+ routing to the host, ev_route), then the
    // resident experts and the shared expert. Host-independent, so it is captured once as a CUDA graph:
    // ~18 launches at ~7 us each on WDDM become one graph launch.
    void enqueue_layer(int l, bool full = false) {
        LayerW& w = L[l];
        rmsnorm(x, l ? moe : nullptr, w.in_ln, h, D, EPS, 1, cs);  // folds in the previous layer's MoE output
        if (is_attn(l)) {
            gemv(w.attn_qkv, h, q, D, AH * 2 * AHD + 2 * AKV * AHD, cs);  // -> q, k, v
            attn_decode(q, k, v, w.qn, w.kn, qo, w.kv, ag, AH, AKV, pos_d, ctx, ROT, THETA, EPS, 1, 0, cs, apart, ASPLIT);
            gemv(w.o, ag, tmp, AH * AHD, D, cs);
        } else {
            gemv(w.dn_in, h, big, D, DC + DVH * DNV + 2 * DVH, cs);  // -> qkv, z, a, b
            deltanet_core(big, z, a, b, DC + DVH * DNV + 2 * DVH, w.conv, w.alog, w.dtb, w.nw, w.ring, w.S, conv, o, DC, CONVK, DVH, DKH, EPS, 1, cs);
            gemv(w.out, o, tmp, DVH * DNV, D, cs);
        }
        rmsnorm(x, tmp, w.post_ln, h, D, EPS, 1, cs);
        gemv_f16(w.router, h, rlog, D, E + 1, cs);  // router logits + the shared expert's gate logit (rlog[E])
        router_topk(rlog, E, TOPK, NORM_TOPK, ridx, rw, cs);
        // hits from the device slot table (the host runs exactly the others: hmask), then the hand-off
        route_hits_k<<<1, 32, 0, cs>>>(ridx, rw, TOPK, l * E, table_d, slots_d, hw_d, nh_d, hmask_hd);
        if (full) export_signal_k<<<1, 256, 0, cs>>>(h, ridx, rw, h_hd, ids_hd, w_hd, D, TOPK, seq_d, rflag_hd + l);
        else {
            export_route_k<<<(D + 255) / 256, 256, 0, cs>>>(h, ridx, rw, h_hd, ids_hd, w_hd, D, TOPK);
            CK(cudaEventRecordWithFlags(ev_route, cs, cudaEventRecordExternal));  // host-visible even from a graph
        }
        // resident experts + shared expert, no host in the loop
        experts_forward(slots_d, hw_d, TOPK, h, act, moe, cs, nh_d);
        shared_expert_add(w.sh_gu, w.sh_d, rlog + E, h, moe, scr, D, SI, cs);
        if (full) wait_add_k<<<1, 256, 0, cs>>>(moe, y_hd, seq_d, dflag_hd + l, D, abort_hd, l, NL, spin_ns, waited_hd);
        // Router-ahead prediction for the own RAM cache's prefetch: after this layer's GPU work, while the GPU waits for
        // the CPU's misses (x is still this layer's pre-MoE residual). The host reads it after the next layer's routing
        // event; double-buffered so the next layer's prediction can't overwrite it first.
        if (ahead_k && l + ahead_k < NL) {
            rmsnorm(x, nullptr, L[l + ahead_k].post_ln, pa_h, D, EPS, 1, cs);
            gemv_f16(L[l + ahead_k].router, pa_h, pa_s, D, E, cs);
            router_topk(pa_s, E, TOPK, NORM_TOPK, pa_idx, pa_w, cs);
            copy_ids_k<<<1, TOPK, 0, cs>>>(pa_idx, pidx_hd + (l & 1) * TOPK);
        }
    }

    std::vector<cudaGraphExec_t> graphs;
    // Whole-step graph (all NL layers, one launch per token; step_full). Not on low-RAM machines (per-layer SSD
    // prefetch, own RAM cache, router-ahead need the host between layers; a page fault would stall a spinning GPU) or MOEX_STEP_GRAPH=0.
    cudaGraphExec_t step_graph = nullptr;
    int *seq_d = nullptr, *rflag_h, *rflag_hd, *dflag_h, *dflag_hd, *abort_h, *abort_hd;
    int seq = 0;
    int* skip_d = nullptr;          // &g_skip
    long long spin_ns = 200000000;  // MOEX_SPIN_MS (default 200): the GPU's longest wait for one layer's CPU result
    long long fallbacks = 0;        // tokens finished on the per-layer path after such a timeout
    long long *waited_h, *waited_hd;  // per layer, ns the GPU waited for the host's CPU result (GPU clock)
    double gpu_wait_s = 0;            // their sum over tokens
    void capture_graphs() {
        if (!low_ram && !rc_n && !ahead_k && !(getenv("MOEX_STEP_GRAPH") && atoi(getenv("MOEX_STEP_GRAPH")) == 0)) {
            CK(cudaMalloc(&seq_d, 4)); CK(cudaMemset(seq_d, 0, 4));
            CK(cudaHostAlloc((void**)&rflag_h, NL * 4, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&rflag_hd, rflag_h, 0));
            CK(cudaHostAlloc((void**)&dflag_h, NL * 4, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&dflag_hd, dflag_h, 0));
            CK(cudaHostAlloc((void**)&waited_h, NL * 8, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&waited_hd, waited_h, 0));
            CK(cudaHostAlloc((void**)&abort_h, 8, cudaHostAllocMapped)); CK(cudaHostGetDevicePointer((void**)&abort_hd, abort_h, 0));
            memset(rflag_h, 0, NL * 4); memset(dflag_h, 0, NL * 4); abort_h[0] = abort_h[1] = 0;
            if (getenv("MOEX_SPIN_MS")) spin_ns = (long long)(atof(getenv("MOEX_SPIN_MS")) * 1e6);
            CK(cudaGetSymbolAddress((void**)&skip_d, g_skip));
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < NL; l++) enqueue_layer(l, true);
            CK(cudaStreamEndCapture(cs, &g));
            CK(cudaGraphInstantiate(&step_graph, g, 0));
            CK(cudaGraphDestroy(g));
        }
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
    static constexpr int CH = 1024;
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
    static constexpr int SB = 24;   // experts per staging batch
    static constexpr int NB = 8;    // staging batches in VRAM: the streamer runs up to NB batches (~1 layer) ahead
    int nring = NB;                 // batches in use: 2 on GPUs whose cache would be small
    static constexpr int STREAM_MIN = 3;  // experts with <= 2 rows are cheaper on the CPU than 1.77 MB over PCIe (sweep: PLAN.md S5)
    uint8_t* stage_p = nullptr;   // VRAM, NB * SB slots
    uint8_t* hstage = nullptr;    // pinned host, 2 * SB slots
    CpuPool cpool{4, false};      // staging copies, concurrent with the CPU experts on cpu.pool; sleeps when idle
    cudaEvent_t ev_hfree[2], ev_dma0[2];  // a pinned half's DMA: start / end (timed: H2D busy time)
    cudaStream_t cpp;
    cudaEvent_t ev_copied[NB], ev_free[NB];
    // Streamer thread (one per prefill): takes expert lists in the order chunk_layer consumes them and stages them
    // batch by batch into the VRAM ring (global batch b -> ring batch b % nring). In chunks of >= AHEAD_T rows, layer
    // l also runs layer l+1's router on its residual (normed with l+1's post norm) and queues the non-resident experts
    // predicted for >= STREAM_MIN rows right after l's routing, so they stream under l's experts and l+1's GEMMs.
    // Real routing decides: unpredicted experts with >= STREAM_MIN rows are queued on demand, the rest go to the CPU.
    static constexpr int AHEAD_T = 256;
    std::thread streamer;
    std::mutex st_mu;
    std::deque<std::pair<int, std::vector<int>>> st_q;  // (layer, experts), guarded by st_mu
    bool st_stop = false;
    std::atomic<long long> st_copied{0}, st_freed{0};  // batches whose H2D copy is enqueued / whose compute is enqueued
    long long main_b = 0;                              // main thread's next batch
    std::vector<int> pred_list;                        // main thread: the next layer's predicted list (queued)
    bool pred_set = false;
    int *predi_h, *predi_d;                            // mapped: predicted top-8 of the next layer, [CH][TOPK]
    double st_wait = 0, st_memcpy = 0, st_dma = 0;     // streamer: waiting for a free pinned half, host memcpy; H2D busy
    long long st_rows[6] = {};                         // streamed experts by the rows they got: 0, 1, 2, 3, 4, 5+
    void stream_push(int l, const std::vector<int>& v) { std::lock_guard<std::mutex> g(st_mu); st_q.push_back({l, v}); }
    void streamer_main() {
        for (long long b = 0;;) {
            std::pair<int, std::vector<int>> job;
            while (true) {
                { std::lock_guard<std::mutex> g(st_mu); if (!st_q.empty()) { job = std::move(st_q.front()); st_q.pop_front(); break; } if (st_stop) return; }
                std::this_thread::yield();
            }
            const int n = (int)job.second.size();
            for (int k0 = 0; k0 < n; k0 += SB, b++) {
                while (b >= nring && st_freed.load() <= b - nring) std::this_thread::yield();  // ring batch b % nring still in use
                const int hb = b & 1, nb = std::min(SB, n - k0);
                uint8_t* hbase = hstage + (size_t)hb * SB * EXPERT_BYTES;
                double q0 = now();
                CK(cudaEventSynchronize(ev_hfree[hb]));  // the DMA out of this pinned half is done
                if (b >= 2) { float ms; CK(cudaEventElapsedTime(&ms, ev_dma0[hb], ev_hfree[hb])); st_dma += ms / 1e3; }
                double q1 = now();
                std::function<void(int)> cp = [&](int i) {
                    int key = job.first * E + job.second[k0 + i];
                    copy_nt(hbase + (size_t)i * EXPERT_BYTES, eq[key], EXPERT_Q_BYTES);
                    copy_nt(hbase + (size_t)i * EXPERT_BYTES + EXPERT_Q_BYTES, es[key], EXPERT_BYTES - EXPERT_Q_BYTES);
                };
                cpool.parallel_for(nb, cp);
                st_wait += q1 - q0; st_memcpy += now() - q1;
                if (b >= nring) CK(cudaStreamWaitEvent(cpp, ev_free[b % nring]));  // its previous batch is computed
                CK(cudaEventRecord(ev_dma0[hb], cpp));
                CK(cudaMemcpyAsync(stage_p + (size_t)(b % nring) * SB * EXPERT_BYTES, hbase, (size_t)nb * EXPERT_BYTES, cudaMemcpyHostToDevice, cpp));
                CK(cudaEventRecord(ev_hfree[hb], cpp));
                CK(cudaEventRecord(ev_copied[b % nring], cpp));
                st_copied.store(b + 1);
            }
        }
    }
    template <class T> static void mapped_alloc(T** hp, T** dp, size_t n) {
        CK(cudaHostAlloc((void**)hp, n * sizeof(T), cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**)dp, *hp, 0));
    }
    // Prefill's device scratch and staging slots live in the last nres slots of the expert arena: during decode
    // they are ordinary cache slots, a prefill evicts them first (measured: as separate allocations they cost
    // ~90 slots and decode 54 -> 52 tok/s). base == nullptr only measures.
    int nres = 0;
    static constexpr int VT = 9;  // rows of the MTP round's scratch: the token + up to 8 drafts
    size_t carve_prefill(uint8_t* base, int ch = CH) {
        const int LQ = AH * 2 * AHD + 2 * AKV * AHD, LB = DC + DVH * DNV + 2 * DVH;
        size_t off = 0;
        auto f = [&](float*& p, size_t n) { if (base) p = (float*)(base + off); off += (n * 4 + 255) & ~(size_t)255; };
        f(Xc, (size_t)ch * D); f(hc, (size_t)ch * D); f(tmpc, (size_t)ch * D); f(moec, (size_t)ch * D);
        f(qkvc, (size_t)ch * LQ); f(bigc, (size_t)ch * LB); f(convc, (size_t)ch * DC); f(oc, (size_t)ch * DVH * DNV);
        f(qoc, (size_t)ch * AH * AHD); f(agc, (size_t)ch * AH * AHD); f(lgc, (size_t)ch * E); f(rwc, (size_t)ch * TOPK);
        f(actp, (size_t)ch * TOPK * EI); f(ypair, (size_t)ch * TOPK * D);
        f(shgu, (size_t)ch * 2 * SI); f(sha, (size_t)ch * SI); f(shs, (size_t)ch * D); f(shl, ch);
        float* ri; f(ri, (size_t)ch * TOPK); if (base) ridxc = (int*)ri;
        if (ch != CH) return off;  // the MTP round streams nothing
        if (base) stage_p = base + off;
        return off + (size_t)nring * SB * EXPERT_BYTES;
    }
    void claim_prefill_slots() {
        while (true) {  // no admission copy may still be on its way into these slots
            { std::lock_guard<std::mutex> g(mu); if (todo.empty()) break; }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        CK(cudaStreamSynchronize(cp));
        poll_admissions();
        for (int s = nslots - nres; s < nslots; s++) {
            if (owner[s] >= 0 && ready[s] && rc_victim) rc_copy_back(owner[s], s, nullptr);
            if (owner[s] >= 0) { slot_of[owner[s]] = -1; if (ready[s]) table_set(owner[s], nullptr); owner[s] = -1; }
            ready[s] = 0; inflight[s] = 1;  // admit() skips them until released
        }
        if (rc_victim) { CK(cudaStreamSynchronize(cp)); rc_poll(); }
        flush_table();
    }
    void release_prefill_slots() {
        for (int s = nslots - nres; s < nslots; s++) { inflight[s] = 0; last[s] = 0; }
    }
    void alloc_prefill() {
        mapped_alloc(&rows_h, &rows_d, (size_t)CH * D); mapped_alloc(&rwc_h, &rwc_d, (size_t)CH * TOPK);
        mapped_alloc(&ycpu_h, &ycpu_d, (size_t)CH * D); mapped_alloc(&ridxc_h, &ridxc_d, (size_t)CH * TOPK);
        mapped_alloc(&items_h, &items_d, (size_t)E); mapped_alloc(&predi_h, &predi_d, (size_t)CH * TOPK); mapped_alloc(&pairs_h, &pairs_d, (size_t)CH * TOPK);
        ypair_cpu.resize((size_t)CH * TOPK * D);
        CK(cudaMallocHost(&hstage, (size_t)2 * SB * EXPERT_BYTES));
        CK(cudaStreamCreateWithFlags(&cpp, cudaStreamNonBlocking));
        for (int i = 0; i < NB; i++) {
            CK(cudaEventCreateWithFlags(&ev_copied[i], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_free[i], cudaEventDisableTiming));
        }
        for (int i = 0; i < 2; i++) {
            CK(cudaEventCreate(&ev_hfree[i])); CK(cudaEventCreate(&ev_dma0[i]));
            CK(cudaEventRecord(ev_hfree[i], cpp));
        }
    }

    // Runs prompt tokens ids[0..n) at positions pos0.. and returns the greedy next token (logits[] filled).
    // Same math as step() except dense projections (fp16 activations on tensor cores) and summation order.
    int prefill(const int* ids, int n, int pos0) {
        const int LQ = AH * 2 * AHD + 2 * AKV * AHD, LB = DC + DVH * DNV + 2 * DVH;
        int next = -1;
        nvtxRangePushA("prefill");
        gate.reset();  // a new prompt: n-gram vs head is judged afresh
        claim_prefill_slots();
        cpool.spin = true;  // the copy pool is hot for the length of the prefill
        double pt0 = now(), pt[5] = {};
        unsigned long pf0 = page_faults();
        long long sn0 = streamed_n;
        std::fill(cl[0], cl[0] + 10, 0.0);
        st_copied = st_freed = 0; main_b = 0; st_stop = false; st_wait = st_memcpy = st_dma = 0;
        std::fill(st_rows, st_rows + 6, 0LL);
        streamer = std::thread([this] { streamer_main(); });
        for (int c0 = 0; c0 < n; c0 += CH) {
            const int T = std::min(CH, n - c0);
            tokens += T;
            for (int t = 0; t < T; t++) note(ids[c0 + t]);
            for (int t = 0; t < T; t++)
                for (int i = 0; i < D; i++) rows_h[(size_t)t * D + i] = __half2float(embed[(size_t)ids[c0 + t] * D + i]);
            copy_k<<<(T * D + 255) / 256, 256, 0, cs>>>(Xc, rows_d, T * D);
            set_int_k<<<1, 1, 0, cs>>>(pos_d, pos0 + c0);
            for (int l = 0; l < NL; l++) chunk_layer(l, T, pt, false);
            if (has_mtp) {  // fill the MTP layer's cache for these positions; the last chunk also makes the first draft
                rmsnorm(Xc, moec, norm, hc, D, EPS, T, cs);  // Xc = final hidden of every row, hc = normed
                calib("lm_head", -1, hc, T, D);
                if (c0 + T == n) { gemv(lm_head, hc + (size_t)(T - 1) * D, logits, D, V, cs); next = pick_token(); }
                draft = mtp_forward(ids + c0 + 1, T, c0 + T < n ? ids[c0 + T] : next, pos0 + c0 + 1, pt, false);
                if (c0 + T == n) choose_draft(draft, next);
            } else if (calib_dir) {  // every row's normed final hidden state: lm_head's calibration input
                rmsnorm(Xc, moec, norm, hc, D, EPS, T, cs);
                calib("lm_head", -1, hc, T, D);
                if (c0 + T == n) { gemv(lm_head, hc + (size_t)(T - 1) * D, logits, D, V, cs); next = pick_token(); }
            } else if (c0 + T == n) {  // the last prompt token: its logits pick the first generated token
                rmsnorm(Xc + (size_t)(T - 1) * D, moec + (size_t)(T - 1) * D, norm, h, D, EPS, 1, cs);
                gemv(lm_head, h, logits, D, V, cs);
                next = pick_token();
            }
        }
        { std::lock_guard<std::mutex> g(st_mu); st_stop = true; }
        streamer.join();
        for (int i = 0; i < 2; i++) { float ms; CK(cudaEventSynchronize(ev_hfree[i])); if (st_copied.load() > i && cudaEventElapsedTime(&ms, ev_dma0[i], ev_hfree[i]) == cudaSuccess) st_dma += ms / 1e3; }
        pt[3] = st_wait; pt[4] = st_memcpy;
        release_prefill_slots();
        cpool.spin = false;
        if (!quiet) fprintf(stderr, "[prefill] %d tok in %.2f s = %.1f tok/s | host: gpu-wait %.2f  streaming-wait %.2f  cpu-experts %.2f s | streamed %lld experts"
                " | copier: wait-dma %.2f  memcpy %.2f s | page faults %lu\n",
                n, now() - pt0, n / (now() - pt0), pt[0], pt[1], pt[2], streamed_n - sn0, pt[3], pt[4], page_faults() - pf0);
        if (!quiet) print_cl(0, 1.0, "s", "prefill chunk-layers, host s");
        if (!quiet) fprintf(stderr, "    streamed %.1f GB, H2D busy %.2f s (%.1f GB/s) | streamed experts by rows used: 0: %lld  1: %lld  2: %lld  3: %lld  4: %lld  5+: %lld\n",
                            (streamed_n - sn0) * (double)EXPERT_BYTES / 1e9, st_dma, (streamed_n - sn0) * (double)EXPERT_BYTES / 1e9 / std::max(st_dma, 1e-9),
                            st_rows[0], st_rows[1], st_rows[2], st_rows[3], st_rows[4], st_rows[5]);
        nvtxRangePop();
        return next;
    }

    // One layer over the T rows in Xc (positions *pos_d..). verify: the MTP round (2 rows, private scratch): no
    // streaming, misses run on the CPU and are admitted to VRAM as in step(); otherwise prefill (streaming).
    void chunk_layer(int l, int T, double* pt, bool verify) {
                const int LQ = AH * 2 * AHD + 2 * AKV * AHD, LB = DC + DVH * DNV + 2 * DVH;
                LayerW& w = L[l];
                if (verify) { poll_admissions(); flush_table(); clock++; }
                double ct = now(), cv[9];  // chunk_layer host timeline (cl[] below)
                auto lap = [&](int i) { double n = now(); cv[i] = n - ct; ct = n; };
                // everything up to the routing (+ the shared expert, which doesn't depend on it). In the MTP round the
                // shapes and buffers are fixed, so it is captured once per (layer, rows) and replayed as one graph launch.
                auto pre = [&] {
                rmsnorm(Xc, l % NL ? moec : nullptr, w.in_ln, hc, D, EPS, T, cs);  // layers 0 and NL (MTP) start fresh
                if (is_attn(l)) {
                    calib("attn_qkv", l, hc, T, D);
                    gemm(w.attn_qkv, hc, qkvc, T, D, LQ, cs);
                    attn_decode(qkvc, qkvc + AH * 2 * AHD, qkvc + AH * 2 * AHD + AKV * AHD, w.qn, w.kn, qoc, w.kv, agc,
                                AH, AKV, pos_d, ctx, ROT, THETA, EPS, T, LQ, cs, verify ? apart : nullptr, verify ? ASPLIT : 1);  // MTP round: split like decode
                    calib("attn_o", l, agc, T, AH * AHD);
                    gemm(w.o, agc, tmpc, T, AH * AHD, D, cs);
                } else {
                    calib("dn_in", l, hc, T, D);
                    gemm(w.dn_in, hc, bigc, T, D, LB, cs);
                    deltanet_core(bigc, bigc + DC, bigc + DC + DVH * DNV, bigc + DC + DVH * DNV + DVH, LB, w.conv, w.alog, w.dtb, w.nw,
                                  w.ring, w.S, convc, oc, DC, CONVK, DVH, DKH, EPS, T, cs,
                                  verify ? snap_S(l) : nullptr, verify ? ring_S(l) : nullptr, RSTRIDE);  // state after row 0, ring after each row
                    if (verify && T > 2) {  // rows 1 .. T-2: the inputs a rollback past row 0 replays
                        CK(cudaMemcpyAsync(rp_c(l), convc + DC, (size_t)(T - 2) * DC * 4, cudaMemcpyDeviceToDevice, cs));
                        CK(cudaMemcpy2DAsync(rp_a(l), 2 * DVH * 4, bigc + LB + DC + DVH * DNV, LB * 4, 2 * DVH * 4, T - 2, cudaMemcpyDeviceToDevice, cs));
                    }
                    calib("dn_out", l, oc, T, DVH * DNV);
                    gemm(w.out, oc, tmpc, T, DVH * DNV, D, cs);
                }
                rmsnorm(Xc, tmpc, w.post_ln, hc, D, EPS, T, cs);
                gemv_f16(w.router, hc, lgc, D, E, cs, T);  // fp32 routing, as decode
                router_topk(lgc, E, TOPK, NORM_TOPK, ridxc, rwc, cs, T);
                export_route_k<<<(T * D + 255) / 256, 256, 0, cs>>>(hc, ridxc, rwc, rows_d, ridxc_d, rwc_d, T * D, T * TOPK);
                if (!verify && T >= AHEAD_T && l + 1 < NL) {  // predict layer l+1's routing (fp16 GEMM: only a prediction)
                    rmsnorm(Xc, nullptr, L[l + 1].post_ln, tmpc, D, EPS, T, cs);
                    Mat r; r.f16 = L[l + 1].router;
                    gemm(r, tmpc, lgc, T, D, E, cs);
                    router_topk(lgc, E, TOPK, NORM_TOPK, predi_d, rwc, cs, T);
                }
                if (verify) {  // the host only needs the routing: signal it, then the shared expert overlaps the CPU work
                    CK(cudaEventRecordWithFlags(ev_route, cs, cudaEventRecordExternal));
                    shared_rows(w, T);
                }
                };
                if (verify) {
                    cudaGraphExec_t& g = vgraph[T - 1][l];
                    if (!g) {
                        vcaptured = true;  // a new row count: this round's time is not representative (DraftGate)
                        cudaGraph_t gr;
                        CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
                        pre();
                        CK(cudaStreamEndCapture(cs, &gr));
                        CK(cudaGraphInstantiate(&g, gr, 0));
                        CK(cudaGraphDestroy(gr));
                    }
                    CK(cudaGraphLaunch(g, cs));
                } else pre();
                lap(0);
                double w0 = now();
                if (verify) CK(cudaEventSynchronize(ev_route)); else CK(cudaStreamSynchronize(cs));
                pt[0] += now() - w0;
                lap(1);
                const double hb0 = now();
                const bool vs_on = verify && T >= 2 && l < NL;  // every draft round (T - 1 drafts)
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
                std::vector<int> streamed, ondemand;  // queued when l-1 predicted l / now, in the streamer's order
                if (!verify && pred_set) { streamed = std::move(pred_list); pred_set = false; }
                char listed[E] = {};
                for (int e : streamed) listed[e] = 1;
                auto add_item = [&](int e, const uint8_t* slot) {
                    items_h[nitems] = {slot, npairs, (int)per[e].size()};
                    for (auto& pr : per[e]) pairs_h[npairs++] = pr;
                    nitems++;
                };
                for (int e = 0; e < E; e++) {
                    if (per[e].empty()) continue;
                    int key = l * E + e, sl = slot_of[key];
                    if (verify && sl >= 0) last[sl] = clock;  // protects this layer's experts (incl. in-flight) from eviction
                    if (vs_on) vs[0]++;
                    if (sl >= 0 && ready[sl]) { add_item(e, arena + (size_t)sl * EXPERT_BYTES); hits += per[e].size(); if (vs_on) vs[1]++; continue; }
                    if (vs_on) { const bool t0 = per[e][0].t == 0; vs[t0 && per[e].size() > 1 ? 4 : t0 ? 2 : 3]++; }  // pairs are in t order
                    misses += per[e].size();
                    if (verify && sl < 0) admits_todo.push_back(key);
                    if (listed[e]) continue;
                    if (!verify && (int)per[e].size() >= STREAM_MIN) { ondemand.push_back(e); continue; }
                    job_of[e] = (int)jobs.size();
                    jobs.push_back({eq[key], es[key], {}, {}, {}});
                    for (auto& pr : per[e]) { jobs.back().toks.push_back(pr.t); jobs.back().pids.push_back(pr.p); jobs.back().ws.push_back(pr.w); }
                }
                if (!ondemand.empty()) stream_push(l, ondemand);
                if (!verify && T >= AHEAD_T && l + 1 < NL) {  // after l's own list: the streamer's order is consumption order
                    int cnt[E] = {};
                    pred_list.clear();
                    for (int i = 0; i < T * TOPK; i++) cnt[predi_h[i]]++;
                    for (int e = 0; e < E; e++) { int sl = slot_of[(l + 1) * E + e]; if (cnt[e] >= STREAM_MIN && !(sl >= 0 && ready[sl])) pred_list.push_back(e); }
                    if (!pred_list.empty()) stream_push(l + 1, pred_list);
                    pred_set = true;
                }
                lap(2);
                {
                    std::vector<int> pk;
                    for (int e = 0; e < E; e++) if (job_of[e] >= 0) pk.push_back(l * E + e);
                    for (int e : streamed) pk.push_back(l * E + e);
                    for (int e : ondemand) pk.push_back(l * E + e);
                    prefetch_experts(pk.data(), (int)pk.size());
                }
                lap(3);
                // GPU: resident experts and the shared expert first (all async)
                CK(cudaMemsetAsync(ypair, 0, (size_t)T * TOPK * D * 4, cs));
                const int nres = nitems;
                // prefill: tensor cores over each expert's rows; the 2-row MTP round: GEMV (each weight read once)
                auto moe_pairs = [&](const PairItem* it, int n) {
                    if (verify || PF_GEMV) pairs_forward(it, n, pairs_d, hc, actp, ypair, cs);
                    else pairs_forward_tc(it, n, pairs_d, hc, actp, ypair, cs);
                };
                moe_pairs(items_d, nres);
                if (!verify) shared_rows(w, T);
                // streamed experts: items point at their ring slot; batch k of this list is global batch b0 + k
                const long long b0 = main_b;
                std::vector<int> bi;  // item index at each batch's start, then the end
                for (auto* v : {&streamed, &ondemand}) {  // two lists: batches don't span lists
                    for (int k = 0; k < (int)v->size(); k++) {
                        const int e = (*v)[k];
                        if (k % SB == 0) bi.push_back(nitems);
                        if (!per[e].empty()) add_item(e, stage_p + ((size_t)((main_b + k / SB) % nring) * SB + k % SB) * EXPERT_BYTES);
                        st_rows[std::min((int)per[e].size(), 5)]++;
                    }
                    main_b += (v->size() + SB - 1) / SB;
                }
                const int nbat = (int)(main_b - b0);
                bi.push_back(nitems);
                int launched = 0;
                auto launch_streamed = [&](bool wait) {  // each batch's experts once its copy is enqueued
                    for (; launched < nbat; launched++) {
                        const long long b = b0 + launched;
                        if (!wait && st_copied.load() <= b) return;
                        while (st_copied.load() <= b) std::this_thread::yield();
                        CK(cudaStreamWaitEvent(cs, ev_copied[b % nring]));
                        if (bi[launched + 1] > bi[launched]) moe_pairs(items_d + bi[launched], bi[launched + 1] - bi[launched]);
                        CK(cudaEventRecord(ev_free[b % nring], cs));
                        st_freed.store(b + 1);
                    }
                };
                launch_streamed(false);
                streamed_n += streamed.size() + ondemand.size();
                lap(4);
                // CPU: non-resident experts, each read once for all of its tokens, overlapped with the above
                double c1 = now();
                if (vs_on) vt[0] += c1 - hb0;
                if (verify) cpu.run_pairs_small(jobs, rows_h, T, ypair_cpu.data());  // the MTP round: int8 kernel, as decode
                else cpu.run_pairs(jobs, rows_h, T, T * TOPK, ypair_cpu.data());
                lap(5);
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
                lap(6);
                double j0 = now();
                launch_streamed(true);
                pt[1] += now() - j0;  // time the main thread waited on the streaming
                lap(7);
                pairs_reduce_k<<<dim3((D + 255) / 256, T), 256, 0, cs>>>(ypair, ycpu_d, moec, TOPK, D);
                gated_add_rows_k<<<dim3((D + 255) / 256, T), 256, 0, cs>>>(moec, shs, shl, D);
                if (verify) { for (int key : admits_todo) admit(key); admits_todo.clear(); start_admissions(); }
                if (vs_on) vt[1] += now() - j0;
                lap(8);
                double* c = cl[verify ? 1 + !jobs.empty() : 0];
                for (int i = 0; i < 9; i++) c[i] += cv[i];
                c[9]++;
    }
    // chunk_layer host time [prefill | MTP round, layer without CPU miss | with]: enqueue up to routing, wait for the
    // routing, group pairs, prefetch, launch resident + shared + copier start, CPU experts, CPU sum, wait copier, tail; count
    double cl[3][10] = {};
    void print_cl(int which, double per, const char* unit, const char* tag) {
        static const char* nm[] = {"enqueue", "gpu-wait", "group", "prefetch", "launch", "cpu-experts", "cpu-sum", "copier-wait", "tail"};
        const double* c = cl[which];
        double tot = 0; for (int i = 0; i < 9; i++) tot += c[i];
        fprintf(stderr, "    %s (%.0f layer runs, %.2f %s):", tag, c[9], tot / per, unit);
        for (int i = 0; i < 9; i++) fprintf(stderr, " %s %.2f", nm[i], c[i] / per);
        fprintf(stderr, "\n");
    }
    // MTP round profile, main layers only: unique experts needed, hits, misses used by the real token only, the draft
    // only, both; host time building the work before the CPU starts, and after it (launches + admissions)
    double vs[5] = {}, vt[2] = {};
    std::vector<int> admits_todo;
    cudaGraphExec_t vgraph[VT][NL + 1] = {};  // MTP round: per (rows - 1, layer), see chunk_layer
    // MOEX_CALIB=dir (research): prefill appends each dense matrix's input rows (fp16) to dir/<group>.L<layer>.f16
    // (lm_head: dir/lm_head.f16), the calibration data of tools/quant_dense.py; MOEX_CALIB_EVERY=n keeps every n-th
    // prompt token (the same rows at every site; ~0.7 MB per kept row over all files). Chunks of <= 2 rows (the MTP
    // round, possibly inside a graph capture) are skipped. Run with --no-mtp: the MTP layer isn't dumped.
    const char* calib_dir = getenv("MOEX_CALIB");
    const int calib_every = getenv("MOEX_CALIB_EVERY") ? std::max(1, atoi(getenv("MOEX_CALIB_EVERY"))) : 1;
    void calib(const char* group, int l, const float* X, int T, int I) {
        if (!calib_dir || T <= 2 || l >= NL) return;
        std::vector<float> x((size_t)T * I);
        CK(cudaMemcpyAsync(x.data(), X, x.size() * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaStreamSynchronize(cs));
        std::vector<half> h;
        for (int t = 0; t < T; t++)  // row t is prompt token tokens - T + t (prefill counts the chunk first)
            if ((tokens - T + t) % calib_every == 0)
                for (int i = 0; i < I; i++) h.push_back(__float2half(x[(size_t)t * I + i]));
        const std::string p = std::string(calib_dir) + "/" + group + (l < 0 ? "" : ".L" + std::to_string(l)) + ".f16";
        FILE* f = fopen(p.c_str(), "ab");
        if (!f || fwrite(h.data(), 2, h.size(), f) != h.size()) { fprintf(stderr, "cannot write %s\n", p.c_str()); exit(1); }
        fclose(f);
    }
    void shared_rows(LayerW& w, int T) {  // shared expert over T rows of hc -> shs, gate logits -> shl
        const int l = int(&w - L.data());
        calib("sh_gu", l, hc, T, D);
        gemm(w.sh_gu, hc, shgu, T, D, 2 * SI, cs);
        silu_mul_rows_k<<<dim3((SI + 255) / 256, T), 256, 0, cs>>>(shgu, sha, SI);
        calib("sh_d", l, sha, T, SI);
        gemm(w.sh_d, sha, shs, T, SI, D, cs);
        gemv_f16(w.sh_gate, hc, shl, D, 1, cs, T);
    }

    // The MTP layer over T rows: row t gets the token at position pos+t (toks[t], the last row `last`) and the main
    // model's final hidden state at pos+t-1 (Xc). Returns the draft for the token after the last row (greedy).
    int mtp_forward(const int* toks, int T, int last, int pos, double* pt, bool verify) {
        for (int t = 0; t < T; t++) {
            const int tok = t < T - 1 ? toks[t] : last;
            for (int i = 0; i < D; i++) rows_h[(size_t)t * D + i] = __half2float(embed[(size_t)tok * D + i]);
        }
        copy_k<<<(T * D + 255) / 256, 256, 0, cs>>>(tmpc, rows_d, T * D);
        rmsnorm(tmpc, nullptr, mtp_ne, hc, D, EPS, T, cs);
        rmsnorm(Xc, nullptr, mtp_nh, tmpc, D, EPS, T, cs);
        CK(cudaMemcpy2DAsync(mtp_cat, 2 * D * 4, hc, D * 4, D * 4, T, cudaMemcpyDeviceToDevice, cs));        // [embedding | hidden]
        CK(cudaMemcpy2DAsync(mtp_cat + D, 2 * D * 4, tmpc, D * 4, D * 4, T, cudaMemcpyDeviceToDevice, cs));
        gemm(mtp_fc, mtp_cat, Xc, T, 2 * D, D, cs);
        set_int_k<<<1, 1, 0, cs>>>(pos_d, pos);
        chunk_layer(NL, T, pt, verify);
        rmsnorm(Xc + (size_t)(T - 1) * D, moec + (size_t)(T - 1) * D, mtp_norm, h, D, EPS, 1, cs);
        gemv(lm_head, h, dlog, D, V, cs);  // own buffer: logits[] holds the real model's last scores
        argmax(dlog, V, amv, ams, ids_hd, cs);
        CK(cudaStreamSynchronize(cs));
        return ids_h[0];
    }

    // Rollback state of the MTP round (written by the kernels as they go): per linear-attention layer, the DeltaNet
    // state after row 0 (snap, 2 MB: the only full copy), the conv ring after every row but the last (rsnap, 96 KB per
    // row), and for rows 1 .. K-1 the recurrence's inputs (rp_c: conv output, rp_a: a | b; ~32 KB per row). Keeping rows
    // 0 .. r: the state after row 0 is copied back, rows 1 .. r are replayed by the same kernel (same order: the same
    // state, bit for bit), the ring after row r copied back. Full per-row states would cost ~66 MB per draft token.
    static constexpr size_t NDN = NL - NL / 4, DNS = (size_t)DVH * DNK * DNV, RSTRIDE = NDN * DC * (CONVK - 1);
    int snap_rows = 1;
    size_t snap_bytes = 0;
    float *rsnap = nullptr, *rp_conv = nullptr, *rp_ab = nullptr, *rp_out = nullptr;
    static size_t dn_of(int l) { return (size_t)(l - (l + 1) / 4); }
    float* snap_S(int l) { return snap + dn_of(l) * DNS; }
    float* ring_S(int l, int r = 0) { return rsnap + r * RSTRIDE + dn_of(l) * DC * (CONVK - 1); }
    float* rp_c(int l) { return rp_conv + dn_of(l) * (snap_rows - 1) * DC; }
    float* rp_a(int l) { return rp_ab + dn_of(l) * (snap_rows - 1) * 2 * DVH; }
    void dn_rollback(int r) {
        for (int l = 0; l < NL; l++) if (!is_attn(l)) {
            CK(cudaMemcpyAsync(L[l].S, snap_S(l), DNS * 4, cudaMemcpyDeviceToDevice, cs));
            CK(cudaMemcpyAsync(L[l].ring, ring_S(l, r), (size_t)DC * (CONVK - 1) * 4, cudaMemcpyDeviceToDevice, cs));
            if (r > 0)  // z = the conv rows (any readable memory: the outputs are not kept)
                dn_recur_k<<<DVH, DNV, 0, cs>>>(rp_c(l), DC, rp_c(l), rp_a(l), rp_a(l) + DVH, 2 * DVH, L[l].alog, L[l].dtb, L[l].nw,
                                                L[l].S, rp_out, DVH, DKH, EPS, r);
        }
        CK(cudaGetLastError());
    }

    // One MTP decode round: feeds y at position p and the K drafts at p+1.. together (K = 1: the head's draft). Returns
    // the new tokens: the accepted drafts (fed), then one more (not yet fed): the residual draw at the first rejection
    // or the token after the last draft.
    long long mtp_rounds = 0, mtp_acc = 0, mtp_drafted = 0;  // rounds, drafts accepted, drafts verified
    double vpt[2] = {};  // in the 40 layers: waiting for the GPU's routing, CPU experts
    double mt[6] = {};  // MTP round time: snapshot, 40 layers, lm_head, mtp layer, rollback (+ replay), pick
    long long picks = 0;  // token_dist calls in rounds
    bool vcaptured = false;  // this round captured a layer graph
    long long rb_n[2] = {};  // rollbacks: to row 0 (copy only), past row 0 (copy + replay)
    double rb_t[2] = {};
    std::vector<int> mtp_round(int y, int p) {
        if (drafts.empty()) set_drafts({draft});
        // snap holds kmax rows; the KV cache ends at ctx
        const int K = std::max(0, std::min({(int)drafts.size(), ngram.nmin ? dlen.kmax : 1, ctx - 1 - p})), T = K + 1;
        drafts.resize(K); dqs.resize(K);
        double pt[5] = {}, t0 = now();
        const double tr = t0;
        vcaptured = false;
        mtp_rounds++;
        nvtxRangePushA("mtp round");
        carve_prefill(vbuf, VT);
        for (int t = 0; t < T; t++) {
            const int tok = t ? drafts[t - 1] : y;
            for (int i = 0; i < D; i++) rows_h[(size_t)t * D + i] = __half2float(embed[(size_t)tok * D + i]);
        }
        copy_k<<<(T * D + 255) / 256, 256, 0, cs>>>(Xc, rows_d, T * D);
        set_int_k<<<1, 1, 0, cs>>>(pos_d, p);
        for (int l = 0; l < NL; l++) chunk_layer(l, T, pt, true);
        CK(cudaStreamSynchronize(cs)); mt[1] += now() - t0; t0 = now();
        vpt[0] += pt[0]; vpt[1] += pt[2];
        rmsnorm(Xc, moec, norm, hc, D, EPS, T, cs);  // Xc = final hidden at p .. p+K
        gemm(lm_head, hc, lg2, T, D, V, cs);
        CK(cudaStreamSynchronize(cs)); mt[2] += now() - t0; t0 = now();
        // greedy: p and q one-hot, a draft is kept iff it is its row's argmax
        auto row_dist = [&](int r) { picks++; float* sv = logits; logits = lg2 + (size_t)r * V; auto d = token_dist(); logits = sv; return d; };
        const std::vector<int> out = spec_verify(drafts, dqs, row_dist, rng);
        mt[5] += now() - t0; t0 = now();
        const int a = (int)out.size() - 1;  // drafts accepted
        mtp_acc += a; mtp_drafted += K;
        if (ng_draft) { ng_rounds++; ng_acc += a; ng_drafted += K; dlen.update(K, a); }
        tokens += a + 1; note(y);
        for (int t = 0; t < a; t++) note(drafts[t]);
        if (a < K) {  // keep rows 0..a (KV rows after p+a hold rejected drafts; they are overwritten when fed)
            dn_rollback(a);
            CK(cudaStreamSynchronize(cs));
            rb_n[a > 0]++; rb_t[a > 0] += now() - t0;
            mt[4] += now() - t0; t0 = now();
        }
        if (!vcaptured) (ng_draft ? gate.ng : gate.head).add(a + 1, deterministic ? 1.0 + 0.1 * (T - 2) : now() - tr);
        const std::vector<int> kept(drafts.begin(), drafts.begin() + a);
        choose_draft(mtp_forward(kept.data(), a + 1, out.back(), p + 1, pt, true), out.back());
        mt[3] += now() - t0;
        carve_prefill(arena + (size_t)(nslots - nres) * EXPERT_BYTES);
        nvtxRangePop();
        return out;
    }

    // ---- token choice: greedy (GPU argmax) or sampled (temperature / top-k / top-p), repetition penalty ----
    struct Sampling { float temp = 0.f, top_p = 1.f, rep = 1.f; int top_k = 0; uint64_t seed = 0; };
    Sampling samp;
    std::mt19937_64 rng;
    std::deque<int> recent;  // last TokList::N tokens of the context, for the repetition penalty
    static constexpr int CAND_CAP = 8192;
    Cand *cand_d = nullptr, *cand_h = nullptr;
    int *cnt_d = nullptr, *cnt_h = nullptr;
    void note(int tok) {
        recent.push_back(tok); if ((int)recent.size() > TokList::N) recent.pop_front();
        if (ngram.nmin) ngram.push(tok);
    }
    // logits[] is being computed on cs; returns the next token.
    int pick_token() {
        std::vector<TokP> p = token_dist();
        return samp.temp <= 0.f ? p[0].id : sample_from(p, rng);
    }
    // The distribution the next token is drawn from: greedy {argmax: 1}; sampled: the candidates top-k / top-p keep,
    // renormalized.
    std::vector<TokP> token_dist() {
        if (samp.rep != 1.f && !recent.empty()) {
            TokList tl;
            for (int t : recent) if (std::find(tl.id, tl.id + tl.n, t) == tl.id + tl.n) tl.id[tl.n++] = t;
            repeat_penalty_k<<<1, TokList::N, 0, cs>>>(logits, tl, samp.rep);
        }
        argmax(logits, V, amv, ams, ids_hd, cs);  // the greedy id lands in mapped host memory; the max in amv[BLOCKS]
        if (samp.temp <= 0.f) { CK(cudaStreamSynchronize(cs)); return {{ids_h[0], 1.0}}; }
        if (!cand_d) {
            CK(cudaMalloc(&cand_d, CAND_CAP * sizeof(Cand))); CK(cudaMallocHost(&cand_h, CAND_CAP * sizeof(Cand)));
            CK(cudaMalloc(&cnt_d, 4)); CK(cudaMallocHost(&cnt_h, 4));
            rng.seed(samp.seed ? samp.seed : (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
        }
        // tokens more than 20 * temp below the top have probability < e^-20 of it: dropping them changes nothing
        CK(cudaMemsetAsync(cnt_d, 0, 4, cs));
        candidates_k<<<ARGMAX_BLOCKS, 256, 0, cs>>>(logits, V, amv + ARGMAX_BLOCKS, 20.f * samp.temp, cand_d, cnt_d, CAND_CAP);
        CK(cudaMemcpyAsync(cnt_h, cnt_d, 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaMemcpyAsync(cand_h, cand_d, CAND_CAP * sizeof(Cand), cudaMemcpyDeviceToHost, cs));
        CK(cudaStreamSynchronize(cs));
        std::vector<Cand> c(cand_h, cand_h + std::min(*cnt_h, CAND_CAP));
        if (*cnt_h > CAND_CAP) {  // a very flat distribution: take the whole vocabulary (rare)
            std::vector<float> lg(V); CK(cudaMemcpy(lg.data(), logits, V * 4, cudaMemcpyDeviceToHost));
            c.clear(); for (int i = 0; i < V; i++) c.push_back({i, lg[i]});
        }
        std::sort(c.begin(), c.end(), [](const Cand& a, const Cand& b) { return a.v > b.v || (a.v == b.v && a.id < b.id); });
        if (samp.top_k > 0 && (int)c.size() > samp.top_k) c.resize(samp.top_k);
        std::vector<double> p(c.size());
        double z = 0; for (size_t i = 0; i < c.size(); i++) z += p[i] = exp((c[i].v - c[0].v) / samp.temp);
        size_t keep = c.size(); double cum = 0;
        for (size_t i = 0; i < c.size(); i++) { cum += p[i] / z; if (cum >= samp.top_p) { keep = i + 1; break; } }
        std::vector<TokP> d(keep);
        for (size_t i = 0; i < keep; i++) d[i] = {c[i].id, p[i] / z / (cum >= samp.top_p ? cum : 1.0)};
        return d;
    }

    // Router-ahead prediction study: layer l+k's router on layer l's residual stream (normed with l+k's post norm),
    // the cheap predictor a layer-ahead SSD prefetch would use. x is still layer l's pre-MoE residual here.
    void write_ahead(int l) {
        constexpr int KA = 5, TOP = 16;
        if (!ahead_d) { CK(cudaMalloc(&ahead_d, (D + KA * E) * 4)); CK(cudaMallocHost(&ahead_h, KA * E * 4)); }
        for (int k = 0; k < KA && l + k < NL; k++) {
            rmsnorm(x, nullptr, L[l + k].post_ln, ahead_d, D, EPS, 1, cs);
            gemv_f16(L[l + k].router, ahead_d, ahead_d + D + k * E, D, E, cs);
        }
        CK(cudaMemcpyAsync(ahead_h, ahead_d + D, KA * E * 4, cudaMemcpyDeviceToHost, cs));
        CK(cudaStreamSynchronize(cs));
        int16_t rec[KA][TOP];
        for (int k = 0; k < KA; k++) {
            int idx[E]; for (int e = 0; e < E; e++) idx[e] = e;
            const float* sc = ahead_h + k * E;
            std::partial_sort(idx, idx + TOP, idx + E, [&](int a, int b) { return sc[a] > sc[b]; });
            for (int j = 0; j < TOP; j++) rec[k][j] = l + k < NL ? (int16_t)idx[j] : -1;
        }
        fwrite(rec, sizeof rec, 1, trace_ahead);
    }

    // step() as one graph launch: the host only polls each layer's routing flag, runs the CPU misses and raises the
    // layer's done flag. Residency is frozen for the token (slot table flushed before the launch, admissions decided
    // after the last layer), so the GPU's hits and the host's misses always partition the top-8.
    std::vector<int> adm_keys;
    bool y_pending = false;  // the last layer's CPU result, in case the GPU timed out before adding it
    int step_full(int tok, int pos, bool want_logits) {
        struct Range { Range() { nvtxRangePushA("decode"); } ~Range() { nvtxRangePop(); } } range;
        tokens++;
        note(tok);
        double h0 = now();
        poll_admissions();
        flush_table();
        for (int i = 0; i < D; i++) h_h[i] = __half2float(embed[(size_t)tok * D + i]);
        copy_k<<<(D + 255) / 256, 256, 0, cs>>>(x, h_hd, D);
        set_int_k<<<1, 1, 0, cs>>>(pos_d, pos);
        set_int_k<<<1, 1, 0, cs>>>(seq_d, ++seq);
        set_int_k<<<1, 1, 0, cs>>>(skip_d, 0);
        CK(cudaGraphLaunch(step_graph, cs));
        ht[0] += now() - h0;
        adm_keys.clear();
        static const int stall_l = getenv("MOEX_TEST_STALL") ? atoi(getenv("MOEX_TEST_STALL")) : -1;  // test: "layer:ms:every"
        static const int stall_ms = getenv("MOEX_TEST_STALL") && strchr(getenv("MOEX_TEST_STALL"), ':') ? atoi(strchr(getenv("MOEX_TEST_STALL"), ':') + 1) : 0;
        static const int stall_every = getenv("MOEX_TEST_STALL") && strrchr(getenv("MOEX_TEST_STALL"), ':') != strchr(getenv("MOEX_TEST_STALL"), ':') ? atoi(strrchr(getenv("MOEX_TEST_STALL"), ':') + 1) : 1;
        for (int l = 0; l < NL; l++) {
            const double w0 = now();
            while (*(volatile int*)&rflag_h[l] != seq) {
                if (*(volatile int*)abort_h == seq) {  // the GPU gave up waiting on layer l - 1: finish on the per-layer path
                    fallbacks++;
                    set_int_k<<<1, 1, 0, cs>>>(skip_d, 0);  // after the rest of the graph, before the per-layer kernels
                    if (y_pending) add_k<<<(D + 255) / 256, 256, 0, cs>>>(moe, y_hd, D);
                    for (int key : adm_keys) if (slot_of[key] < 0) admit(key);
                    start_admissions();
                    return step(tok, pos, want_logits, l);
                }
                if (now() - w0 > 10.0) { CK(cudaStreamSynchronize(cs)); fprintf(stderr, "decode: layer %d's routing never arrived\n", l); exit(1); }
                _mm_pause();
            }
            const double w1 = now();
            ht[1] += w1 - w0;
            clock++;
            int nm = 0;
            const uint8_t* mq[TOPK]; const float* msc[TOPK]; float mw[TOPK];
            for (int j = 0; j < TOPK; j++) {
                int key = l * E + ids_h[j], sl = slot_of[key];
                touch(key);
                if (sl >= 0) last[sl] = clock;
                if (*(volatile unsigned*)hmask_h >> j & 1) hits++;
                else { mw[nm] = w_h[j]; mq[nm] = eq[key]; msc[nm++] = es[key]; misses++; if (sl < 0) adm_keys.push_back(key); }
            }
            if (nm) {
                cpu.run(nm, mq, msc, mw, h_h, y_h);
                cpu_s += now() - w1; ht[2] += now() - w1; hs[1][2] += now() - w1;
            }
            if (l == stall_l && stall_ms && tokens % stall_every == 0) std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms));
            std::atomic_thread_fence(std::memory_order_seq_cst);  // y_h before the flag
            *(volatile int*)&dflag_h[l] = 2 * seq + (nm > 0);
            y_pending = nm > 0;
            hs[nm > 0][1] += w1 - w0; hs[nm > 0][4]++;
            // Admit this layer's misses now (the GPU already has its flag): the loader's slow mmap -> pinned copies start
            // half a token earlier than at the token's end (hit rate 60% -> per-layer path's ~67%). Safe mid-graph: the
            // H2D copies wait for this graph (after_compute is recorded behind it on cs), a victim's table entry is
            // cleared on cs before the next graph, and the host follows the GPU's hit mask, so a victim the GPU still
            // takes later in this token stays consistent. Deterministic mode admits at the token's end (its poll waits
            // for every copy, which would wait for this graph).
            if (!deterministic && !adm_keys.empty()) {
                const double a1 = now();
                poll_admissions();
                for (int key : adm_keys) if (slot_of[key] < 0) admit(key);
                adm_keys.clear();
                start_admissions();
                ht[3] += now() - a1;
            }
        }
        // the last layer's wait: passed, or timed out (then its CPU result is added here)
        for (const double w0 = now(); ((volatile int*)abort_h)[1] != seq && ((volatile int*)abort_h)[0] != seq; _mm_pause())
            if (now() - w0 > 10.0) { CK(cudaStreamSynchronize(cs)); fprintf(stderr, "decode: the step graph never finished\n"); exit(1); }
        if (((volatile int*)abort_h)[0] == seq) { fallbacks++; set_int_k<<<1, 1, 0, cs>>>(skip_d, 0); if (y_pending) add_k<<<(D + 255) / 256, 256, 0, cs>>>(moe, y_hd, D); }
        else for (int l = 0; l < NL; l++) gpu_wait_s += ((volatile long long*)waited_h)[l] * 1e-9;
        rmsnorm(x, moe, norm, h, D, EPS, 1, cs);
        if (want_logits) gemv(lm_head, h, logits, D, V, cs);
        const double a0 = now();  // admissions overlap the lm_head; their copies wait for this token's graph
        poll_admissions();  // frees the stages of finished copies (their table updates land before the next graph)
        for (int key : adm_keys) if (slot_of[key] < 0) admit(key);
        start_admissions();
        ht[3] += now() - a0;
        return want_logits ? pick_token() : -1;
    }

    // from > 0: the rest of a token whose whole-step graph stopped before layer `from` (step_full)
    int step(int tok, int pos, bool want_logits, int from = 0) {
        if (!from && step_graph && !prof && !trace && !trace_ahead) return step_full(tok, pos, want_logits);
        struct Range { Range() { nvtxRangePushA("decode"); } ~Range() { nvtxRangePop(); } } range;
        double t = now();
        if (!from) {
            tokens++;
            note(tok);
            for (int i = 0; i < D; i++) h_h[i] = __half2float(embed[(size_t)tok * D + i]);
            copy_k<<<(D + 255) / 256, 256, 0, cs>>>(x, h_hd, D);  // zero-copy from the mapped host row
            set_int_k<<<1, 1, 0, cs>>>(pos_d, pos);
        }
        for (int l = from; l < NL; l++) {
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
            if (trace_ahead) write_ahead(l);
            clock++;
            int nm = 0, mkey[TOPK], mall[TOPK];
            const uint8_t* mq[TOPK]; const float* msc[TOPK]; float mw[TOPK];
            for (int j = 0; j < TOPK; j++) {
                int key = l * E + ids_h[j], sl = slot_of[key];
                touch(key);
                if (sl >= 0) last[sl] = clock;  // protects this step's experts (incl. in-flight) from eviction
                if (*(volatile unsigned*)hmask_h >> j & 1) hits++;
                else { mw[nm] = w_h[j]; mall[nm] = key; mkey[nm++] = sl < 0 ? key : -1; misses++; }
            }
            if (rc_n) { rc_poll(); rc_prefetch(l); rc_need(mall, nm); }
            else prefetch_experts(mall, nm);
            for (int j = 0; j < nm; j++) { mq[j] = eq[mall[j]]; msc[j] = es[mall[j]]; }
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
                hs[1][2] += c1 - c0; hs[1][3] += now() - c1;
            }
            hs[nm > 0][0] += h1 - h0; hs[nm > 0][1] += h2 - h1; hs[nm > 0][4]++;
            mark(1, t);
        }
        const float* pending = moe;
        // the last MoE output is folded into x here, or by the next token's first norm
        rmsnorm(x, pending, norm, h, D, EPS, 1, cs);
        if (!want_logits) return -1;
        gemv(lm_head, h, logits, D, V, cs);
        int next = pick_token();
        mark(2, t);
        return next;
    }
};

#include "session.h"
#include "serve.h"
#include "download.h"
#include "bench.h"

// One line of user input as UTF-8 (the Windows console delivers non-ASCII input correctly only as UTF-16).
static bool read_line(std::string& out) {
#ifndef _WIN32
    if (!std::getline(std::cin, out)) return false;
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
#else
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (!GetConsoleMode(in, &mode)) { if (!std::getline(std::cin, out)) return false; if (!out.empty() && out.back() == '\r') out.pop_back(); return true; }
    std::wstring w; wchar_t buf[512]; DWORD got;
    while (true) {
        if (!ReadConsoleW(in, buf, 512, &got, nullptr) || got == 0) return false;
        w.append(buf, got);
        if (!w.empty() && w.back() == L'\n') break;
    }
    while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    out.assign(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), out.data(), n, nullptr, nullptr);
    return true;
#endif
}

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

// The CPU expert kernels are AVX2 (every x86 CPU since ~2013); say so instead of crashing on an illegal instruction.
static bool has_avx2() {
#ifndef _WIN32
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    int r[4];
    __cpuid(r, 1);
    bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
    if (!osxsave || !avx || !fma || (_xgetbv(0) & 6) != 6) return false;
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;
#endif
}

int main(int argc, char** argv) {
    if (!has_avx2()) { fprintf(stderr, "moex needs a CPU with AVX2 and FMA (Intel Haswell / AMD Zen or newer).\n"); return 1; }
    std::string model = default_model_dir(), only;
    std::string mode, system_prompt;
    bool think = false, samp_set = false, n_set = false, warm = false;
    int mtp = -1;  // --mtp 1, --no-mtp 0, else on where it was measured faster
    std::string host = "127.0.0.1";
    std::vector<std::string> allow_origins;  // serve: web pages (Origin) besides localhost ones that may call
    int port = 8765;
    if (argc > 1 && argv[1][0] != '-') { mode = argv[1]; argv++; argc--; }  // moex chat | moex serve
    std::vector<int> ids, stop;
    int n = 32, ctx = 2048;
    bool ctx_set = false;  // --ctx given: chat/serve/bench keep it, even 2048
    std::string check, batch, trace_path, ahead_path;
    bool prof = false, f16 = false, use_prefill = true;
    std::string q4groups, q6groups;
    Engine::Sampling samp;  // default greedy
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", s.c_str()); exit(2); } return std::string(argv[++i]); };
        if (s == "--ids") ids = parse_ids(next());
        else if (s == "--n") { n = std::stoi(next()); n_set = true; }
        else if (s == "--model") model = next();
        else if (s == "--only") only = next();
        else if (s == "--system") system_prompt = next();
        else if (s == "--think") think = true;
        else if (s == "--load") warm = true;
        else if (s == "--mtp") mtp = 1;
        else if (s == "--no-mtp") mtp = 0;
        else if (s == "-v" || s == "--version") mode = "version";
        else if (s == "--host") host = next();
        else if (s == "--allow-origin") allow_origins.push_back(next());
        else if (s == "--port") port = std::stoi(next());
        else if (s == "--ctx") { ctx = std::stoi(next()); ctx_set = true; }
        else if (s == "--stop") stop = parse_ids(next());
        else if (s == "--check") check = next();
        else if (s == "--profile") prof = true;
        else if (s == "--f16") f16 = true;
        else if (s == "--no-prefill") use_prefill = false;
        else if (s == "--q4") q4groups = next();
        else if (s == "--q6") q6groups = next();
        else if (s == "--temp") { samp.temp = std::stof(next()); samp_set = true; }
        else if (s == "--top-k") samp.top_k = std::stoi(next());
        else if (s == "--top-p") samp.top_p = std::stof(next());
        else if (s == "--repeat-penalty") samp.rep = std::stof(next());
        else if (s == "--seed") samp.seed = std::stoull(next());
        else if (s == "--batch") batch = next();
        else if (s == "--trace") trace_path = next();
        else if (s == "--trace-ahead") ahead_path = next();
        else if (s == "-h" || s == "--help") mode = "help";
        else { fprintf(stderr, "unknown argument: %s\n\n", s.c_str()); mode = "badarg"; break; }
    }
    if (!check.empty()) {  // ids come from the golden run's meta.json
        FILE* f = fopen((check + "/meta.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s/meta.json\n", check.c_str()); return 1; }
        std::string m(1 << 16, '\0'); m.resize(fread(m.data(), 1, m.size(), f)); fclose(f);
        size_t a = m.find('[', m.find("\"ids\"")) + 1, b = m.find(']', a);
        ids = parse_ids(m.substr(a, b - a));
    }
    if (mode == "version") { printf("moex %s\n", VERSION); return 0; }
    const bool bad = !mode.empty() && mode != "help" && mode != "download" && mode != "chat" && mode != "serve" && mode != "install" && mode != "bench";
    if (bad && mode != "badarg") fprintf(stderr, "unknown command: %s\n\n", mode.c_str());
    if (bad || mode == "help" || (mode.empty() && ids.empty() && batch.empty())) {  // unknown input: the help page, exit 2
        fprintf(mode == "help" ? stdout : stderr,
                "moex %s - Qwen3.6-35B-A3B on a consumer NVIDIA GPU\n\n"
                "Commands:\n"
                "  moex download          download the model (~22 GB, resumable: run again if it stops)\n"
                "  moex chat              chat in this terminal (/reset new conversation, /exit quit)\n"
                "  moex serve             OpenAI-compatible API on http://127.0.0.1:8765/v1\n"
                "  moex bench             measure speed on this PC (~2 min), with a line to share and tips\n"
                "  moex install           put moex on your PATH, so `moex` works in any terminal\n"
                "  moex help | -h | --help, moex --version\n\n"
                "Options:\n"
                "  --load                 load everything before the prompt (~30 s), then every reply is full speed\n"
                "  --system \"TEXT\"        system prompt (chat)\n"
                "  --think                let the model reason before answering\n"
                "  --temp T --top-p P --top-k K --repeat-penalty R --seed N   sampling (chat: 0.7 / 0.8 / 20)\n"
                "  --ctx N                context length in tokens (chat/serve default 8192)\n"
                "  --port N --host H      server address (serve; default 8765, 127.0.0.1 = this PC only)\n"
                "  --allow-origin URL     serve: let web pages from URL call moex (default: localhost pages only; * = any)\n"
                "  --model DIR            model folder (default %s; or set MOEX_MODEL)\n"
                "  --no-mtp               no multi-token prediction (on by default when RAM holds every expert: ~5%% faster)\n\n"
                "Developer: --ids 1,2,3 [--n N] | --batch ids.txt | --check ref/eval1 | --f16 | --profile\n",
                VERSION, model.c_str());
        return mode == "help" ? 0 : 2;
    }
    if (mode == "install") return install_moex();
    if (mode == "serve") {  // before the 20 s load: a mistyped --host fails right away
        uint8_t ip[4];
        if (!parse_host(host, ip)) { fprintf(stderr, "--host %s is not an IPv4 address (e.g. 127.0.0.1, or 0.0.0.0 to open moex to your network)\n", host.c_str()); return 2; }
    }
    if ((mode == "chat" || mode == "bench") && !samp_set) { samp.temp = 0.7f; samp.top_p = 0.8f; samp.top_k = 20; }  // Qwen non-thinking defaults
    if (mode == "chat" && think && !samp_set) { samp.temp = 1.0f; samp.top_p = 0.95f; samp.top_k = 20; }  // thinking defaults
    if ((mode == "chat" || mode == "serve" || mode == "bench") && !ctx_set) ctx = 8192;
    if (mode == "bench") { warm = true; if (!samp.seed) samp.seed = 1; }  // what `moex chat --load` gets, same text every run
    if (mode == "download") return download_model(model, only);
    if (!std::filesystem::exists(std::filesystem::path(model) / "config.json")) {
        fprintf(stderr, "No model at %s.\nRun `moex download` first (or pass --model DIR).\n", model.c_str());
        return 1;
    }

    {  // the only thing moex needs besides itself: an NVIDIA GPU with a recent driver
        int ndev = 0;
        cudaError_t e = cudaGetDeviceCount(&ndev);
        if (e == cudaErrorInsufficientDriver || e == cudaErrorNoDevice || ndev == 0) {
            fprintf(stderr, e == cudaErrorInsufficientDriver
                ? (CUDART_VERSION >= 13000
                    ? "Your NVIDIA driver is too old for this moex (it needs driver 580 or newer). Either install the latest driver\n"
                      "(free, ~700 MB): https://www.nvidia.com/Download/index.aspx\n"
                      "or use the CUDA 12 build, which runs on older drivers: moex-windows-x64-cuda12.zip from\n"
                      "  https://github.com/neerajdad123-byte/moe/releases/latest\n"
                    : "Your NVIDIA driver is too old for moex. Install the latest one (free, ~700 MB):\n  https://www.nvidia.com/Download/index.aspx\n")
                : "moex needs an NVIDIA GPU (RTX 20 series or newer, 6 GB+ VRAM) with its driver installed:\n  https://www.nvidia.com/Download/index.aspx\n");
            return 1;
        }
    }
    const bool interactive = mode == "chat" || mode == "serve" || mode == "bench";
    if (interactive) mem_report("before load");
    Engine eng(model.c_str(), ctx);
    const bool mtp_head = std::filesystem::exists(std::filesystem::path(model) / "mtp.safetensors");
    eng.f16 = f16; eng.q4groups = q4groups; eng.samp = samp;
    if (!q6groups.empty()) eng.q6groups = q6groups;
    eng.want_mtp = mtp == 1 || (mtp < 0 && interactive && mtp_head && !low_ram_machine());
    eng.warm = warm || interactive;
    eng.quiet = interactive;
    eng.pin_async = interactive && !warm;  // --load waits for every expert to be in RAM before the prompt
    eng.hot_file = model + "/moex_hot.bin";
    eng.load_hot({eng.hot_file, exe_dir() + "/moex_hot.bin"});  // counts accumulate across sessions
    eng.load();
    if (interactive) mem_report("after load");
    if (interactive && mtp < 0 && !mtp_head && !eng.low_ram)  // every expert fits in RAM: the case where MTP was measured faster
        fprintf(stderr, "[tip] run `moex download` once more to get the MTP head: multi-token prediction is ~5%% faster on this PC\n");

    if (mode == "bench") {
        Tokenizer tk(model + "/tokenizer.json");
        return run_bench(eng, tk, model);
    }
    if (mode == "serve") {
        Tokenizer tk(model + "/tokenizer.json");
        Session ss(eng, tk);
        ss.think = think;  // the default for requests that don't say (chat_template_kwargs.enable_thinking, reasoning_effort)
        serve(ss, tk, host, port, "qwen3.6-35b-a3b", allow_origins);
        return 0;
    }
    if (mode == "chat") {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
#endif
        Tokenizer tk(model + "/tokenizer.json");
        Session ss(eng, tk);
        ss.think = think;
        std::vector<ChatMsg> msgs;
        if (!system_prompt.empty()) msgs.push_back({"system", system_prompt});
        const int max_new = n_set ? n : 2048;
        fprintf(stderr, warm ? "\nNow the stage is yours. (moex chat - Qwen3.6-35B-A3B. /reset starts over, /exit quits.)\n"
                             : "\nmoex chat - Qwen3.6-35B-A3B. /reset starts over, /exit quits.\n"
                               "(experts are still loading into RAM for ~20 s: the first reply can be slower; `moex chat --load` waits instead)\n");
        std::string line;
        while (true) {
            fprintf(stdout, "\n> "); fflush(stdout);
            if (!read_line(line) || line == "/exit") break;
            if (line.empty()) continue;
            if (line == "/reset") { msgs.resize(system_prompt.empty() ? 0 : 1); ss.ctx.clear(); fprintf(stderr, "(new conversation)\n"); continue; }
            msgs.push_back({"user", line});
            auto prompt = ss.render(msgs);
            if ((int)prompt.size() + 1 >= eng.ctx) {  // generate() would return nothing; an empty reply in msgs would break every later turn
                msgs.pop_back();
                fprintf(stderr, "(the conversation is longer than --ctx %d: /reset starts a new one, or restart with a larger --ctx)\n", eng.ctx);
                continue;
            }
            Utf8Stream us; std::string reply;
            double t0 = now(), t1 = 0; int ntok = 0;
            if (ss.think) { fputs("<think>\n", stdout); fflush(stdout); }  // the prompt ends with it (Qwen3.6 template)
            auto out = ss.generate(prompt, max_new, [&](int t) {
                if (!ntok++) t1 = now();
                std::string txt = us.push(tk.piece(t));
                fputs(txt.c_str(), stdout); fflush(stdout);
                reply += txt;
                return true;
            });
            fputs(us.held.c_str(), stdout); fflush(stdout);  // an incomplete UTF-8 sequence at the end (cut off by --n)
            reply += us.held;
            ss.remember(reply, out);
            eng.save_hot();
            msgs.push_back({"assistant", reply});
            double t2 = now();
            fprintf(stderr, "\n[%d tokens, %.1f tok/s, first token %.2f s, context %zu]\n", ntok,
                    ntok > 1 ? (ntok - 1) / (t2 - t1) : 0.0, t1 - t0, ss.ctx.size());
        }
        return 0;
    }

    if (!check.empty()) {
        int pos = 0, tok = ids[0], bad = 0, scored = 0; double worst = 0, worst_kl = 0, sum_kl = 0;
        int np = (int)ids.size();  // teacher forcing: every id in meta.json is fed; the prompt part is prefilled
        {
            FILE* f = fopen((check + "/meta.json").c_str(), "rb");
            std::string m(1 << 20, 0); m.resize(fread(m.data(), 1, m.size(), f)); fclose(f);
            size_t k = m.find("\"n_prompt\"");
            if (k != std::string::npos) np = std::stoi(m.substr(m.find(':', k) + 1));
        }
        if (use_prefill) pos = np - 1;  // the prompt runs as one prefill; compare from its last position on
        // MTP: the logits of an accepted round's later rows, for the next positions (pend_pos, pend_pos + 1, ..)
        std::vector<std::vector<float>> pend; std::vector<int> pend_id;
        int pend_pos = -1;
        // teacher-forced drafts per round: 1 (the head's slot), or MOEX_NGRAM_MAX with MOEX_NGRAM set (multi-token rounds)
        const int KCHK = eng.ngram.nmin ? eng.dlen.kmax : 1;
        // MOEX_CHECK_DUMP=1: write this run's logits as the reference files (with --f16: a reference for new prompts)
        const bool dump = getenv("MOEX_CHECK_DUMP") != nullptr;
        for (;; pos++) {
            const std::string rpath = check + "/p" + std::string(pos < 10 ? "00" : pos < 100 ? "0" : "") + std::to_string(pos) + "_logits.bin";
            auto ref = dump ? std::vector<float>() : read_f32(rpath);
            if (dump ? pos >= (int)ids.size() : ref.empty()) break;
            std::vector<float> got(V);
            int id;
            if (pend_pos >= 0 && pos - pend_pos < (int)pend.size()) { got = pend[pos - pend_pos]; id = pend_id[pos - pend_pos]; }
            else if (eng.has_mtp && pos > np - 1 && pos + 1 < (int)ids.size()) {
                // MTP rounds, teacher-forced: the true next tokens are the drafts, so every row up to the model's first
                // disagreement is checked; a disagreement exercises the rollback
                {
                    std::vector<int> dr(ids.begin() + pos + 1, ids.begin() + std::min(pos + 1 + KCHK, (int)ids.size()));
                    // MOEX_CHECK_WRONG=j: draft j is made wrong every round, so each round rolls back to row j (replays rows 1..j)
                    if (const char* w = getenv("MOEX_CHECK_WRONG")) if (atoi(w) < (int)dr.size()) dr[atoi(w)] = (dr[atoi(w)] + 1) % V;
                    eng.set_drafts(dr);
                    auto ts = eng.mtp_round(tok, pos);
                    CK(cudaMemcpy(got.data(), eng.lg2, V * 4, cudaMemcpyDeviceToHost));
                    id = ts[0];
                    pend.assign(ts.size() - 1, std::vector<float>(V)); pend_id.assign(ts.begin() + 1, ts.end()); pend_pos = pos + 1;
                    for (size_t r = 1; r < ts.size(); r++) CK(cudaMemcpy(pend[r - 1].data(), eng.lg2 + r * V, V * 4, cudaMemcpyDeviceToHost));
                }
            } else {
                id = use_prefill && pos == np - 1 ? eng.prefill(ids.data(), np, 0) : eng.step(tok, pos, true);
                CK(cudaMemcpy(got.data(), eng.logits, V * 4, cudaMemcpyDeviceToHost));
            }
            if (dump) { ref = got; FILE* fo = fopen(rpath.c_str(), "wb"); if (!fo) { fprintf(stderr, "cannot write %s\n", rpath.c_str()); return 1; } fwrite(ref.data(), 4, V, fo); fclose(fo); }
            double mx = 0, err = 0;
            for (int i = 0; i < V; i++) { mx = std::max(mx, (double)fabsf(ref[i])); err = std::max(err, (double)fabsf(got[i] - ref[i])); }
            int rid = int(std::max_element(ref.begin(), ref.end()) - ref.begin());
            // KL(ref || got) of the next-token distributions
            double mr = ref[rid], mg = *std::max_element(got.begin(), got.end()), zr = 0, zg = 0, kl = 0;
            for (int i = 0; i < V; i++) { zr += exp(ref[i] - mr); zg += exp(got[i] - mg); }
            for (int i = 0; i < V; i++) { double lp = ref[i] - mr - log(zr), lq = got[i] - mg - log(zg); kl += exp(lp) * (lp - lq); }
            worst = std::max(worst, err / mx); worst_kl = std::max(worst_kl, kl); bad += id != rid;
            sum_kl += kl; scored++;
            fprintf(stderr, "  pos %2d tok %6d  logits rel %.3e  KL %.2e  argmax %6d ref %6d %s\n", pos, tok, err / mx, kl, id, rid, id == rid ? "" : "MISMATCH");
            tok = pos + 1 < (int)ids.size() ? ids[pos + 1] : id;
        }
        // Judged statistically: mean KL and top-1 agreement (a quantized model legitimately flips a near-tie now and
        // then). The KV cache is int8 in every mode, so F16 dense weights are no longer exact: measured mean KL ~5e-4
        // from the KV cache alone (it was 7e-13 with an fp32 cache), held to a 5x tighter bar than int8 weights.
        const double mean_kl = scored ? sum_kl / scored : 1e9, agree = scored ? 1.0 - (double)bad / scored : 0;
        bool ok = scored > 0 && (f16 ? mean_kl < 1e-3 && agree >= 0.97 : mean_kl < 5e-3 && agree >= 0.97);
        fprintf(stderr, "[check] %s, %d positions, mean KL %.2e, worst KL %.2e, worst logits rel %.3e, top-1 agreement %.1f%% (%d mismatch) -> %s\n",
                f16 ? "F16 (int8 KV)" : q4groups.empty() ? "int8" : ("int8 + int4 " + q4groups).c_str(), scored, mean_kl, worst_kl, worst,
                100 * agree, bad, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    struct Stats { int dec = 0; double sec = 0; long long hits = 0, misses = 0, admits = 0; double cpu = 0, ht[4] = {}, ttft = 0, gwait = 0; };
    // Prompt token-by-token, then up to n greedy tokens. Returns decode-phase stats.
    auto generate = [&](const std::vector<int>& p, std::vector<int>& out, bool stream) {
        Stats st;
        if ((int)p.size() + n > ctx) { fprintf(stderr, "prompt + n exceeds --ctx %d\n", ctx); exit(2); }
        eng.trace_phase = 0;
        double t0 = now();
        int pos = 0, tok = -1;
        if (use_prefill && !eng.trace) { tok = eng.prefill(p.data(), (int)p.size(), 0); pos = (int)p.size(); }
        else for (; pos < (int)p.size(); pos++) tok = eng.step(p[pos], pos, pos + 1 == (int)p.size());
        long long h0 = eng.hits, m0 = eng.misses, a0 = eng.admits; double c0 = eng.cpu_s, gw0 = eng.gpu_wait_s, t1 = now();
        double ht0[4]; for (int i = 0; i < 4; i++) ht0[i] = eng.ht[i];
        eng.trace_phase = 1;
        const bool mtp = eng.has_mtp && use_prefill && !eng.trace;
        auto emit = [&](int t) {  // false: stop
            out.push_back(t);
            if (stream) { printf("%d\n", t); fflush(stdout); }
            return std::find(stop.begin(), stop.end(), t) == stop.end() && (int)out.size() < n;
        };
        for (bool more = emit(tok); more;) {
            if (mtp) {
                auto ts = eng.mtp_round(tok, pos);
                pos += (int)ts.size();
                for (int t : ts) { st.dec++; if (!(more = emit(t))) break; }
                tok = ts.back();
            } else {
                tok = eng.step(tok, pos++, true);
                st.dec++;
                more = emit(tok);
            }
        }
        st.ttft = t1 - t0;
        st.sec = now() - t1; st.hits = eng.hits - h0; st.misses = eng.misses - m0; st.admits = eng.admits - a0; st.cpu = eng.cpu_s - c0; st.gwait = eng.gpu_wait_s - gw0;
        for (int i = 0; i < 4; i++) st.ht[i] = eng.ht[i] - ht0[i];
        return st;
    };
    auto report = [&](const char* tag, const Stats& st) {
        int dec = std::max(st.dec, 1);
        fprintf(stderr, "[%s] %d tok in %.2f s = %.2f tok/s | expert hit %.1f%% | %.1f misses/tok on CPU = %.1f ms/tok | %lld admitted\n",
                tag, st.dec, st.sec, st.dec / std::max(st.sec, 1e-9), 100.0 * st.hits / std::max(st.hits + st.misses, 1LL),
                (double)st.misses / dec, 1e3 * st.cpu / dec, st.admits);
        fprintf(stderr, "    host ms/tok: launch %.2f  wait-routing %.2f  cpu-misses %.2f  admissions %.2f | TTFT %.2f s | step graph %s, fallbacks %lld, GPU waited on CPU %.2f ms/tok\n",
                1e3 * st.ht[0] / dec, 1e3 * st.ht[1] / dec, 1e3 * st.ht[2] / dec, 1e3 * st.ht[3] / dec, st.ttft, eng.step_graph ? "on" : "off", eng.fallbacks, 1e3 * st.gwait / dec);
    };
    if (!trace_path.empty() && !(eng.trace = fopen(trace_path.c_str(), "wb"))) { fprintf(stderr, "cannot write %s\n", trace_path.c_str()); return 1; }
    if (!ahead_path.empty() && (trace_path.empty() || !(eng.trace_ahead = fopen(ahead_path.c_str(), "wb")))) {
        fprintf(stderr, "--trace-ahead needs --trace and a writable %s\n", ahead_path.c_str()); return 1;
    }

    if (!batch.empty()) {
        FILE* f = fopen(batch.c_str(), "rb");
        if (!f) { fprintf(stderr, "no %s\n", batch.c_str()); return 1; }
        std::vector<std::vector<int>> prompts;
        std::vector<char> line(1 << 24);  // one prompt per line, up to ~2M tokens
        while (fgets(line.data(), (int)line.size(), f)) { auto v = parse_ids(line.data()); if (!v.empty()) prompts.push_back(v); }
        fclose(f);
        Stats tot;
        for (size_t i = 0; i < prompts.size(); i++) {
            eng.reset();
            eng.trace_prompt = (int)i;
            const long long mr0 = eng.mtp_rounds, ma0 = eng.mtp_acc, md0 = eng.mtp_drafted;
            std::vector<int> out;
            Stats st = generate(prompts[i], out, false);
            for (size_t j = 0; j < out.size(); j++) printf(j ? ",%d" : "%d", out[j]);
            printf("\n"); fflush(stdout);
            char tag[32]; snprintf(tag, sizeof tag, "prompt %zu", i);
            report(tag, st);
            if (eng.mtp_rounds > mr0)
                fprintf(stderr, "    mtp: %lld rounds, draft accepted %.1f%%, %.2f tokens/round\n", eng.mtp_rounds - mr0,
                        100.0 * (eng.mtp_acc - ma0) / (eng.mtp_drafted - md0), 1.0 + (double)(eng.mtp_acc - ma0) / (eng.mtp_rounds - mr0));
            if (eng.ngram.nmin) fprintf(stderr, "    n-gram drafts (run so far): %lld rounds (%.1f%%), %.2f drafts/round, accepted %.1f%%, draft length now %d,"
                                        " gated off %lld | this prompt, tokens per ms: n-gram %.4f (%d rounds), head %.4f (%d)\n",
                                        eng.ng_rounds, 100.0 * eng.ng_rounds / std::max(1LL, eng.mtp_rounds), (double)eng.ng_drafted / std::max(1LL, eng.ng_rounds),
                                        100.0 * eng.ng_acc / std::max(1LL, eng.ng_drafted), eng.dlen.k(), eng.ng_gated,
                                        eng.gate.ng.r() / 1e3, eng.gate.ng.n, eng.gate.head.r() / 1e3, eng.gate.head.n);
            tot.dec += st.dec; tot.sec += st.sec; tot.hits += st.hits; tot.misses += st.misses; tot.admits += st.admits; tot.cpu += st.cpu; tot.gwait += st.gwait;
            for (int k = 0; k < 4; k++) tot.ht[k] += st.ht[k];
        }
        report("batch total", tot);
        for (int m = 0; m < 2; m++) if (eng.hs[m][4]) {
            const double* s = eng.hs[m]; const double nl = s[4];
            fprintf(stderr, "    decode layers %s: %.0f (%.1f%%) | us/layer: launch %.0f  wait-routing %.0f  cpu-misses %.0f  admissions %.0f\n",
                    m ? "with a CPU miss " : "all hits in VRAM", nl, 100 * nl / (eng.hs[0][4] + eng.hs[1][4]),
                    1e6 * s[0] / nl, 1e6 * s[1] / nl, 1e6 * s[2] / nl, 1e6 * s[3] / nl);
        }
        for (int m = 1; m < 3; m++) if (eng.cl[m][9]) eng.print_cl(m, eng.cl[m][9] / 1e6, "us/layer", m == 1 ? "MTP layers, all hits in VRAM" : "MTP layers, with a CPU miss ");
        eng.rc_report(tot.dec);
        if (eng.cpu.calls) {
            const CpuExperts& c = eng.cpu; const double t = std::max(tot.dec, 1);
            fprintf(stderr, "    cpu experts (%s): %.1f calls/tok, %.1f items/call | wall ms/tok: quantize x %.2f  gate+up phase %.2f  down phase %.2f"
                            " | item CPU ms/tok: gate+up %.2f  act+requant %.2f  down %.2f\n", c.vnni ? "AVX-VNNI" : "AVX2",
                    c.calls / t, (double)c.items / c.calls, 1e3 * c.t_quant / t, 1e3 * c.t_par / t, 1e3 * c.t_red / t,
                    c.ns_gu / 1e6 / t, c.ns_act / 1e6 / t, c.ns_dn / 1e6 / t);
        }
        if (eng.mtp_rounds) {
            const double r = (double)eng.mtp_rounds;
            fprintf(stderr, "[mtp] %lld rounds, %.2f drafts/round, accepted %.1f%% | ms/round: 40 layers %.2f  lm_head %.2f  pick %.2f (%.3f per row, %.2f rows)"
                            "  mtp layer %.2f  rollback %.2f | layers: gpu-wait %.2f  cpu-experts %.2f\n",
                    eng.mtp_rounds, eng.mtp_drafted / r, 100.0 * eng.mtp_acc / eng.mtp_drafted, 1e3 * eng.mt[1] / r, 1e3 * eng.mt[2] / r,
                    1e3 * eng.mt[5] / r, 1e3 * eng.mt[5] / std::max(1LL, eng.picks), eng.picks / r, 1e3 * eng.mt[3] / r, 1e3 * eng.mt[4] / r,
                    1e3 * eng.vpt[0] / r, 1e3 * eng.vpt[1] / r);
            fprintf(stderr, "[mtp] rollbacks: to row 0 %lld x %.3f ms (state copy) | past row 0 %lld x %.3f ms (state copy + replay)\n",
                    eng.rb_n[0], 1e3 * eng.rb_t[0] / std::max(1LL, eng.rb_n[0]), eng.rb_n[1], 1e3 * eng.rb_t[1] / std::max(1LL, eng.rb_n[1]));
            const double* v = eng.vs;
            fprintf(stderr, "[mtp] per round, 40 layers: %.0f experts needed, %.0f in VRAM | CPU misses: real token only %.0f, draft only %.0f, both %.0f"
                            " | host: before CPU %.2f ms, after CPU %.2f ms\n",
                    v[0] / r, v[1] / r, v[2] / r, v[3] / r, v[4] / r, 1e3 * eng.vt[0] / r, 1e3 * eng.vt[1] / r);
        }
        if (getenv("MOEX_SAVE_HOT")) { eng.hot_file = getenv("MOEX_SAVE_HOT"); eng.save_hot(); }  // makes the shipped moex_hot.bin
        if (eng.trace) fclose(eng.trace);
        if (eng.trace_ahead) fclose(eng.trace_ahead);
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
