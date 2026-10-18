// O1: CPU routed-expert FFN for cache misses (AVX2, fp32 math, int4 gs64 weights read from the mmap).
#pragma once
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

// Persistent worker pool. Workers spin between jobs: a decode token dispatches ~80 jobs and
// waking threads through the OS costs tens of microseconds each on Windows. Once no job has come for HOT_MS
// (generation ended: the chat prompt, a server between requests) they block on gen (WaitOnAddress) and take
// no CPU; the next job wakes them, paying the OS wake-up once.
// Every worker joins every job and reports when it has left it, so the next job's counters are only
// reset once no thread can still be grabbing items of the previous one.
struct CpuPool {
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> next{0}, finished{0};
    std::atomic<bool> quit{false};
    std::atomic<int> sleepers{0};  // workers blocked in gen.wait: only then does a new job pay for a wake-up call
    const std::function<void(int)>* fn = nullptr;
    int n = 0;

    // spin = false: idle workers block right after the short pause loop instead of yielding for HOT_MS (for a
    // pool used only now and then, so its idle threads don't take cores from the hot pool)
    std::atomic<bool> spin;  // may be switched while running (a pool that is hot only now and then)
    static constexpr int HOT_MS = 50;  // well above the gaps between decode tokens
    explicit CpuPool(int threads, bool spin_ = true) : spin(spin_) {
        for (int i = 0; i < threads - 1; i++) th.emplace_back([this] { worker(); });  // caller is the last worker
    }
    ~CpuPool() { quit = true; gen++; gen.notify_all(); for (auto& t : th) t.join(); }

    void run_items() {
        for (int i; (i = next.fetch_add(1)) < n;) (*fn)(i);
    }
    void worker() {
        uint64_t seen = 0;
        while (true) {
            wait_job(seen);
            seen++;  // main never starts job g+1 before every worker finished job g
            if (quit) return;
            run_items();
            finished.fetch_add(1, std::memory_order_release);
        }
    }
    // Returns once gen != seen: pause-spins (~1 ms), then yields while hot, then sleeps until notified.
    void wait_job(uint64_t seen) {
        for (int spins = 0; spins < 20000; spins++) {
            if (gen.load(std::memory_order_acquire) != seen) return;
            _mm_pause();
        }
        const auto t0 = std::chrono::steady_clock::now();
        while (spin && std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(HOT_MS)) {
            if (gen.load(std::memory_order_acquire) != seen) return;
            std::this_thread::yield();
        }
        // seq_cst on both sides (sleepers here, gen in parallel_for): either this wait sees the new gen or
        // parallel_for sees the sleeper and notifies, so no wake-up is lost
        sleepers.fetch_add(1);
        gen.wait(seen);
        sleepers.fetch_sub(1);
    }
    // Calls f(i) for i in [0, items), spread over the pool; returns when all are done.
    void parallel_for(int items, const std::function<void(int)>& f) {
        fn = &f; n = items; next = 0; finished = 0;
        gen.fetch_add(1);
        if (sleepers.load()) gen.notify_all();
        run_items();
        while (finished.load(std::memory_order_acquire) < (int)th.size()) _mm_pause();
    }
};

// 64 int4 weights (32 bytes, one scale group) . x, with x split into even (xe) / odd (xo) elements:
// byte k holds element 2k in the low nibble and 2k+1 in the high nibble.
static inline __m256 dot_group(const uint8_t* q, const float* xe, const float* xo, __m256 acc, float scale) {
    __m256i b = _mm256_loadu_si256((const __m256i*)q);
    const __m256i m4 = _mm256_set1_epi8(0x0F), s8 = _mm256_set1_epi8(8);
    __m256i lo = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(b, m4), s8), s8);                       // signed [-8,7]
    __m256i hi = _mm256_sub_epi8(_mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m4), s8), s8);
    __m256 g = _mm256_setzero_ps();
    alignas(32) int8_t L[32], H[32];
    _mm256_store_si256((__m256i*)L, lo); _mm256_store_si256((__m256i*)H, hi);
    for (int c = 0; c < 4; c++) {
        __m256 fl = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(L + 8 * c))));
        __m256 fh = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(H + 8 * c))));
        g = _mm256_fmadd_ps(fl, _mm256_loadu_ps(xe + 8 * c), g);
        g = _mm256_fmadd_ps(fh, _mm256_loadu_ps(xo + 8 * c), g);
    }
    return _mm256_fmadd_ps(g, _mm256_set1_ps(scale), acc);
}

static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Row dot over I inputs: I/64 groups, scales sc[I/64].
static inline float row_dot(const uint8_t* q, const float* sc, const float* xe, const float* xo, int I) {
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < I / 64; g++) acc = dot_group(q + 32 * g, xe + 32 * g, xo + 32 * g, acc, sc[g]);
    return hsum(acc);
}

static void deinterleave(const float* x, float* xe, float* xo, int n) {
    for (int i = 0; i < n / 2; i++) { xe[i] = x[2 * i]; xo[i] = x[2 * i + 1]; }
}

// out[H] = sum_j w[j] * down_j(silu(gate_j x) * up_j x) for m experts given as (quantized blob, scales).
// Geometry: H=2048, I=512, gs=64 (same as the GPU kernels).
struct CpuExperts {
    static constexpr int H = 2048, I = 512, GS = 64;
    CpuPool pool;
    std::vector<float> xe, xo, act, ae, ao;
    explicit CpuExperts(int threads) : pool(threads), xe(H / 2), xo(H / 2), act(8 * I), ae(8 * I / 2), ao(8 * I / 2) {}

    void run(int m, const uint8_t* const* q, const float* const* sc, const float* w, const float* x, float* out) {
        deinterleave(x, xe.data(), xo.data(), H);
        constexpr int RB = 32;  // rows per work item
        std::function<void(int)> gu = [&](int it) {
            int j = it / (I / RB), r0 = (it % (I / RB)) * RB;
            for (int r = r0; r < r0 + RB; r++) {
                float g = row_dot(q[j] + (size_t)r * (H / 2), sc[j] + r * (H / GS), xe.data(), xo.data(), H);
                float u = row_dot(q[j] + (size_t)(I + r) * (H / 2), sc[j] + I * (H / GS) + r * (H / GS), xe.data(), xo.data(), H);
                act[j * I + r] = g / (1.f + expf(-g)) * u;
            }
        };
        pool.parallel_for(m * (I / RB), gu);
        for (int j = 0; j < m; j++) deinterleave(act.data() + j * I, ae.data() + j * I / 2, ao.data() + j * I / 2, I);
        constexpr int DB = 64;
        std::function<void(int)> dn = [&](int it) {
            for (int d = it * DB; d < it * DB + DB; d++) {
                float acc = 0.f;
                for (int j = 0; j < m; j++)
                    acc += w[j] * row_dot(q[j] + (size_t)2 * I * (H / 2) + (size_t)d * (I / 2),
                                          sc[j] + 2 * I * (H / GS) + d * (I / GS), ae.data() + j * I / 2, ao.data() + j * I / 2, I);
                out[d] = acc;
            }
        };
        pool.parallel_for(H / DB, dn);
    }

    // Prefill: many (token, expert) pairs grouped by expert, so each expert's weights are read once for all
    // of its tokens. X: [T][H] rows; pair outputs ypair[pid][H] = w * expert(x_t). All threads share one
    // parallel_for over (expert, row block) items.
    struct Job { const uint8_t* q; const float* sc; std::vector<int> toks, pids; std::vector<float> ws; };
    std::vector<float> pxe, pxo, pact, pae, pao;
    void run_pairs(const std::vector<Job>& jobs, const float* X, int T, int npairs, float* ypair) {
        if (jobs.empty()) return;
        pxe.resize((size_t)T * H / 2); pxo.resize((size_t)T * H / 2);
        for (int t = 0; t < T; t++) deinterleave(X + (size_t)t * H, pxe.data() + (size_t)t * H / 2, pxo.data() + (size_t)t * H / 2, H);
        pact.resize((size_t)npairs * I); pae.resize((size_t)npairs * I / 2); pao.resize((size_t)npairs * I / 2);
        constexpr int RB = 32, DB = 64;
        const int nj = (int)jobs.size();
        std::function<void(int)> gu = [&](int it) {
            const Job& jb = jobs[it / (I / RB)];
            int r0 = (it % (I / RB)) * RB;
            for (int r = r0; r < r0 + RB; r++)
                for (size_t k = 0; k < jb.toks.size(); k++) {
                    const float *xe = pxe.data() + (size_t)jb.toks[k] * H / 2, *xo = pxo.data() + (size_t)jb.toks[k] * H / 2;
                    float g = row_dot(jb.q + (size_t)r * (H / 2), jb.sc + r * (H / GS), xe, xo, H);
                    float u = row_dot(jb.q + (size_t)(I + r) * (H / 2), jb.sc + I * (H / GS) + r * (H / GS), xe, xo, H);
                    pact[(size_t)jb.pids[k] * I + r] = g / (1.f + expf(-g)) * u;
                }
        };
        pool.parallel_for(nj * (I / RB), gu);
        std::function<void(int)> di = [&](int j) {
            for (int p : jobs[j].pids) deinterleave(pact.data() + (size_t)p * I, pae.data() + (size_t)p * I / 2, pao.data() + (size_t)p * I / 2, I);
        };
        pool.parallel_for(nj, di);
        std::function<void(int)> dn = [&](int it) {
            const Job& jb = jobs[it / (H / DB)];
            int d0 = (it % (H / DB)) * DB;
            for (int d = d0; d < d0 + DB; d++)
                for (size_t k = 0; k < jb.toks.size(); k++) {
                    int p = jb.pids[k];
                    ypair[(size_t)p * H + d] = jb.ws[k] * row_dot(jb.q + (size_t)2 * I * (H / 2) + (size_t)d * (I / 2),
                                                                  jb.sc + 2 * I * (H / GS) + d * (I / GS),
                                                                  pae.data() + (size_t)p * I / 2, pao.data() + (size_t)p * I / 2, I);
                }
        };
        pool.parallel_for(nj * (H / DB), dn);
    }
};
