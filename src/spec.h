// Speculative decoding helpers, host only (tests/test_spec.cpp): a prompt-lookup n-gram drafter and the p/q
// acceptance rule of speculative sampling.
#pragma once
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

struct TokP { int id; double p; };  // one entry of a sparse distribution

inline double prob_of(const std::vector<TokP>& v, int id) {
    for (const TokP& e : v) if (e.id == id) return e.p;
    return 0.0;
}

inline int sample_from(const std::vector<TokP>& v, std::mt19937_64& rng) {
    double z = 0; for (const TokP& e : v) z += e.p;
    double u = std::uniform_real_distribution<double>(0.0, z)(rng), acc = 0;
    for (const TokP& e : v) { acc += e.p; if (u < acc) return e.id; }
    return v.back().id;
}

// Prompt lookup: the longest suffix of the context (nmax down to nmin tokens) that occurred earlier in it; q = how
// often each token followed those occurrences (sorted by count, the most recent first on ties). nmin = 0: off.
// shortcut: a linear scan per draft (~0.1 ms at 26K tokens vs a ~38 ms round); a suffix index if contexts grow 10x.
struct NgramDrafter {
    int nmin = 0, nmax = 4;
    std::vector<int> h;  // every token fed so far
    void reset() { h.clear(); }
    void push(int t) { h.push_back(t); }
    // tail: the tokens after h (the one picked but not yet fed, then drafts so far); q is for the token after them
    std::vector<TokP> propose(const std::vector<int>& tail) {
        h.insert(h.end(), tail.begin(), tail.end());
        std::vector<TokP> q;
        const int n = (int)h.size();
        for (int k = std::min(nmax, n - 1); nmin > 0 && k >= nmin && q.empty(); k--) {
            double tot = 0;
            for (int i = n - k - 1; i >= 0; i--) {  // h[i, i+k) == the last k tokens, followed by h[i+k]
                if (!std::equal(h.begin() + i, h.begin() + i + k, h.end() - k)) continue;
                auto it = std::find_if(q.begin(), q.end(), [&](const TokP& e) { return e.id == h[i + k]; });
                if (it == q.end()) q.push_back({h[i + k], 1.0}); else it->p += 1.0;
                tot++;
            }
            for (TokP& e : q) e.p /= tot;
            std::stable_sort(q.begin(), q.end(), [](const TokP& a, const TokP& b) { return a.p > b.p; });
        }
        h.resize(h.size() - tail.size());
        return q;
    }
    // Up to k drafts after `last`, each proposed given the drafts before it, stopping at the first position without a
    // match. greedy: the most frequent continuation, q one-hot; else drawn from q. qs[i] = the distribution of d[i].
    void chain(int last, int k, bool greedy, std::mt19937_64& rng, std::vector<int>& d, std::vector<std::vector<TokP>>& qs) {
        d.clear(); qs.clear();
        std::vector<int> tail{last};
        while ((int)d.size() < k) {
            std::vector<TokP> q = propose(tail);
            if (q.empty()) break;
            const int t = greedy ? q[0].id : sample_from(q, rng);
            d.push_back(t); tail.push_back(t);
            qs.push_back(greedy ? std::vector<TokP>{{t, 1.0}} : q);
        }
    }
};

// Adaptive draft length from recent acceptance: alpha = EMA of the per-token acceptance rate; a run of accepted drafts
// is ~alpha / (1 - alpha) long, so draft one past it, within [1, kmax].
struct DraftLen {
    int kmax = 4;
    double alpha = 0.75;
    int k() const { return std::max(1, std::min(kmax, 1 + (int)(alpha / std::max(1.0 - alpha, 1e-3)))); }
    void update(int drafted, int accepted) { if (drafted > 0) alpha = 0.8 * alpha + 0.2 * accepted / drafted; }
};

// Speculative sampling (Leviathan et al. 2023, Chen et al. 2023): the draft d was drawn from q, p is the target's
// distribution at that position. Accept d with probability min(1, p(d) / q(d)); otherwise draw from
// norm(max(0, p - q)). The token returned is distributed exactly as p, whatever q is. Returns {accepted, token}.
inline std::pair<bool, int> spec_accept(const std::vector<TokP>& p, const std::vector<TokP>& q, int d, std::mt19937_64& rng) {
    const double pd = prob_of(p, d), qd = prob_of(q, d);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (qd > 0 && U(rng) * qd < pd) return {true, d};
    std::vector<TokP> r;
    for (const TokP& e : p) { const double v = e.p - prob_of(q, e.id); if (v > 0) r.push_back({e.id, v}); }
    if (r.empty()) return {false, sample_from(p, rng)};  // p <= q everywhere means p == q: only reached by rounding
    return {false, sample_from(r, rng)};
}

// A round of k drafts: dist(i) = the target's distribution for the position of d[i] (i = k: the token after the last
// draft), given the tokens before it, which are the drafts while they are accepted. Returns the tokens kept: the
// accepted drafts, then one more (the residual draw at the first rejection, or a draw from dist(k)).
template <class Dist>
std::vector<int> spec_verify(const std::vector<int>& d, const std::vector<std::vector<TokP>>& qs, Dist dist, std::mt19937_64& rng) {
    std::vector<int> out;
    for (size_t i = 0; i < d.size(); i++) {
        auto [ok, t] = spec_accept(dist((int)i), qs[i], d[i], rng);
        out.push_back(t);
        if (!ok) return out;
    }
    out.push_back(sample_from(dist((int)d.size()), rng));
    return out;
}
