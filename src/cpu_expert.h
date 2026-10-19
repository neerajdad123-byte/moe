// O1: CPU routed-expert FFN for cache misses (int4 gs64 weights read from the mmap). Decode: int8 activations,
// AVX-VNNI or AVX2 integer dots; prefill (run_pairs): fp32 math.
#pragma once
#include <immintrin.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <algorithm>
#include <cmath>
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

// ---- decode path: int8 activations (per 64-group, deinterleaved like the nibbles) x int4 weights, integer dots ----
// A nibble v holds w = v - 16 * (v >= 8), so v ^ 8 = w + 8 is unsigned: sum (w+8) x - 8 sum x = sum w x, with the
// 8 * sum x of each group precomputed. AVX-VNNI (vpdpbusd) when the CPU has it, else AVX2 maddubs + madd.
static bool cpu_has_avxvnni() {
    int r[4];
#ifdef _WIN32
    __cpuidex(r, 7, 1);
#else
    __cpuid_count(7, 1, r[0], r[1], r[2], r[3]);
#endif
    return (r[0] >> 4) & 1;
}
#if defined(_MSC_VER) || defined(__AVXVNNI__)
#define MOEX_VNNI 1
#endif
template <bool VNNI>
static inline __m256i dot_u8s8(__m256i acc, __m256i u, __m256i s) {
#ifdef MOEX_VNNI
    if constexpr (VNNI) return _mm256_dpbusd_avx_epi32(acc, u, s);
#endif
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));  // |15 * 127 * 2| < 2^15
}
// One 64-group (32 nibble bytes) against its int8 activations: 8 int32 partial sums.
template <bool VNNI>
static inline __m256i group_i8(const uint8_t* q, const int8_t* xe, const int8_t* xo) {
    const __m256i b = _mm256_loadu_si256((const __m256i*)q), m4 = _mm256_set1_epi8(0x0F), s8 = _mm256_set1_epi8(8);
    const __m256i lo = _mm256_xor_si256(_mm256_and_si256(b, m4), s8), hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(b, 4), m4), s8);
    __m256i i = dot_u8s8<VNNI>(_mm256_setzero_si256(), lo, _mm256_load_si256((const __m256i*)xe));
    return dot_u8s8<VNNI>(i, hi, _mm256_load_si256((const __m256i*)xo));
}
// Quantized activations: groups of 64 -> 32 even + 32 odd int8, scale s, c = 8 * s * sum(q).
struct QAct {
    std::vector<int8_t> buf;
    int8_t *e, *o; float *s, *c;
    void resize(int n) {  // e, o 32-byte aligned
        buf.resize(n + 64 + (n / 64) * 8 * 2 + 64);
        int8_t* b = (int8_t*)(((uintptr_t)buf.data() + 31) & ~(uintptr_t)31);
        e = b; o = b + n / 2; s = (float*)(b + n); c = s + n / 64;
    }
};
static inline void quant_groups(const float* x, int n, int8_t* e, int8_t* o, float* s, float* c) {
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int g = 0; g < n / 64; g++) {
        const float* xg = x + 64 * g;
        __m256 v[8], mx = _mm256_setzero_ps();
        for (int k = 0; k < 8; k++) { v[k] = _mm256_loadu_ps(xg + 8 * k); mx = _mm256_max_ps(mx, _mm256_and_ps(v[k], absm)); }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4)); m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float m = _mm_cvtss_f32(m4), sc = m > 0.f ? m / 127.f : 1.f;
        const __m256 inv = _mm256_set1_ps(1.f / sc);
        alignas(32) int32_t t[64];
        __m256i sum = _mm256_setzero_si256();
        for (int k = 0; k < 8; k++) {
            const __m256i iv = _mm256_cvtps_epi32(_mm256_mul_ps(v[k], inv));  // round to nearest even
            _mm256_store_si256((__m256i*)(t + 8 * k), iv);
            sum = _mm256_add_epi32(sum, iv);
        }
        for (int k = 0; k < 32; k++) { e[32 * g + k] = (int8_t)t[2 * k]; o[32 * g + k] = (int8_t)t[2 * k + 1]; }
        alignas(32) int32_t ss[8]; _mm256_store_si256((__m256i*)ss, sum);
        s[g] = sc; c[g] = 8.f * sc * (ss[0] + ss[1] + ss[2] + ss[3] + ss[4] + ss[5] + ss[6] + ss[7]);
    }
}
// Row dot over `groups` 64-groups: weights q (nibbles), weight scales ws[groups].
template <bool VNNI>
static inline float row_dot_i8(const uint8_t* q, const float* ws, const QAct& x, int groups) {
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.f;
    for (int g = 0; g < groups; g++) {
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(group_i8<VNNI>(q + 32 * g, x.e + 32 * g, x.o + 32 * g)), _mm256_set1_ps(ws[g] * x.s[g]), acc);
        corr += ws[g] * x.c[g];
    }
    return hsum(acc) - corr;
}

// out[H] = sum_j w[j] * down_j(silu(gate_j x) * up_j x) for m experts given as (quantized blob, scales).
// Geometry: H=2048, I=512, gs=64 (same as the GPU kernels).
struct CpuExperts {
    static constexpr int H = 2048, I = 512, GS = 64;
    CpuPool pool;
    explicit CpuExperts(int threads) : pool(threads) {}

    // Decode: int8 activations, integer dots, two phases. (1) item = (expert j, 64-row block g): its gate/up rows, then
    // their activations quantized as group g of expert j's down input. (2) item = 64 output rows: each row's whole down
    // row (256 contiguous bytes, 8 groups) for every expert. A fused layout (each item one column group of down for all
    // rows) measured slower: two items share each cache line of down, so down's RAM traffic doubled (PLAN.md D3).
    const bool vnni = cpu_has_avxvnni();
    // phase profile (decode run): wall s of quantize x / gate+up phase / down phase; item CPU ns: gate+up, act+requant, down
    double t_quant = 0, t_par = 0, t_red = 0;
    long long calls = 0, items = 0;
    std::atomic<long long> ns_gu{0}, ns_act{0}, ns_dn{0};
    QAct qx, qa[8];  // x; each expert's down input
    void run(int m, const uint8_t* const* q, const float* const* sc, const float* w, const float* x, float* out) {
        if (vnni) run_i8<true>(m, q, sc, w, x, out); else run_i8<false>(m, q, sc, w, x, out);
    }
    template <bool VNNI>
    void run_i8(int m, const uint8_t* const* q, const float* const* sc, const float* w, const float* x, float* out) {
        constexpr int NG = I / GS, DB = 64;  // gate/up items per expert; down rows per item
        if (qx.buf.empty()) { qx.resize(H); for (auto& a : qa) a.resize(I); }
        using clk = std::chrono::steady_clock;
        auto sec = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
        auto ns = [](clk::time_point a, clk::time_point b) { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
        const auto c0 = clk::now();
        quant_groups(x, H, qx.e, qx.o, qx.s, qx.c);
        const auto c1 = clk::now();
        std::function<void(int)> gu = [&](int it) {
            const int j = it / NG, g = it % NG, r0 = g * GS;
            const auto i0 = clk::now();
            alignas(32) float a[GS];
            for (int r = 0; r < GS; r++) {
                const float gt = row_dot_i8<VNNI>(q[j] + (size_t)(r0 + r) * (H / 2), sc[j] + (r0 + r) * (H / GS), qx, H / GS);
                const float u = row_dot_i8<VNNI>(q[j] + (size_t)(I + r0 + r) * (H / 2), sc[j] + (I + r0 + r) * (H / GS), qx, H / GS);
                a[r] = w[j] * gt / (1.f + expf(-gt)) * u;  // the routing weight folded in before quantizing
            }
            const auto i1 = clk::now();
            quant_groups(a, GS, qa[j].e + 32 * g, qa[j].o + 32 * g, qa[j].s + g, qa[j].c + g);
            const auto i2 = clk::now();
            ns_gu += ns(i0, i1); ns_act += ns(i1, i2);
        };
        pool.parallel_for(m * NG, gu);
        const auto c2 = clk::now();
        std::function<void(int)> dn = [&](int it) {
            const auto i0 = clk::now();
            for (int d = it * DB; d < it * DB + DB; d++) {
                float acc = 0.f;
                for (int j = 0; j < m; j++)
                    acc += row_dot_i8<VNNI>(q[j] + (size_t)2 * I * (H / 2) + (size_t)d * (I / 2), sc[j] + 2 * I * (H / GS) + d * (I / GS), qa[j], I / GS);
                out[d] = acc;
            }
            ns_dn += ns(i0, clk::now());
        };
        pool.parallel_for(H / DB, dn);
        const auto c3 = clk::now();
        t_quant += sec(c0, c1); t_par += sec(c1, c2); t_red += sec(c2, c3); calls++; items += m * NG + H / DB;
    }

    // Prefill: many (token, expert) pairs grouped by expert, so each expert's weights are read once for all
    // of its tokens. X: [T][H] rows; pair outputs ypair[pid][H] = w * expert(x_t). All threads share one
    // parallel_for over (expert, row block) items.
    struct Job { const uint8_t* q; const float* sc; std::vector<int> toks, pids; std::vector<float> ws; };
    // The MTP round's CPU misses (T <= 9 rows): run_i8's two phases over (expert, token) pairs; each weight row is read
    // once and dotted with every row that uses the expert. ypair[pid][H] = w * expert(x_t), as run_pairs.
    std::vector<QAct> qxt, qp;
    void run_pairs_small(const std::vector<Job>& jobs, const float* X, int T, float* ypair) {
        if (vnni) run_pairs_i8<true>(jobs, X, T, ypair); else run_pairs_i8<false>(jobs, X, T, ypair);
    }
    template <bool VNNI>
    void run_pairs_i8(const std::vector<Job>& jobs, const float* X, int T, float* ypair) {
        if (jobs.empty()) return;
        constexpr int NG = I / GS, DB = 64;
        if ((int)qxt.size() < T) { qxt.resize(T); for (auto& a : qxt) if (a.buf.empty()) a.resize(H); }
        int np = 0; for (auto& jb : jobs) for (int p : jb.pids) np = std::max(np, p + 1);
        if ((int)qp.size() < np) { qp.resize(np); for (auto& a : qp) if (a.buf.empty()) a.resize(I); }
        for (int t = 0; t < T; t++) quant_groups(X + (size_t)t * H, H, qxt[t].e, qxt[t].o, qxt[t].s, qxt[t].c);
        const int nj = (int)jobs.size();
        std::function<void(int)> gu = [&](int it) {
            const Job& jb = jobs[it / NG];
            const int g = it % NG, r0 = g * GS, nk = (int)jb.toks.size();
            alignas(32) float a[9][GS];
            for (int r = 0; r < GS; r++)
                for (int k = 0; k < nk; k++) {
                    const QAct& x = qxt[jb.toks[k]];
                    const float gt = row_dot_i8<VNNI>(jb.q + (size_t)(r0 + r) * (H / 2), jb.sc + (r0 + r) * (H / GS), x, H / GS);
                    const float u = row_dot_i8<VNNI>(jb.q + (size_t)(I + r0 + r) * (H / 2), jb.sc + (I + r0 + r) * (H / GS), x, H / GS);
                    a[k][r] = jb.ws[k] * gt / (1.f + expf(-gt)) * u;
                }
            for (int k = 0; k < nk; k++) { QAct& d = qp[jb.pids[k]]; quant_groups(a[k], GS, d.e + 32 * g, d.o + 32 * g, d.s + g, d.c + g); }
        };
        pool.parallel_for(nj * NG, gu);
        std::function<void(int)> dn = [&](int it) {
            for (const Job& jb : jobs)
                for (int d = it * DB; d < it * DB + DB; d++)
                    for (int p : jb.pids)
                        ypair[(size_t)p * H + d] = row_dot_i8<VNNI>(jb.q + (size_t)2 * I * (H / 2) + (size_t)d * (I / 2), jb.sc + 2 * I * (H / GS) + d * (I / GS), qp[p], I / GS);
        };
        pool.parallel_for(H / DB, dn);
    }

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
