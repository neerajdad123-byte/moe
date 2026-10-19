// CPU unit test of src/spec.h: the prompt-lookup drafter and p/q speculative sampling (output distributed as p,
// acceptance rate = sum min(p, q)). build.bat -> build\test_spec.exe
#include <cmath>
#include <cstdio>
#include "../src/spec.h"

static int fails = 0;
static void check(bool ok, const char* what) { printf("%s %s\n", ok ? "ok  " : "FAIL", what); fails += !ok; }

static NgramDrafter drafter(std::vector<int> h, int nmin, int nmax = 4) {
    NgramDrafter d; d.nmin = nmin; d.nmax = nmax;
    for (int t : h) d.push(t);
    return d;
}

int main() {
    {  // longest suffix 1 2 3 4 occurred twice: followed by 9, then (more recent) by 7
        auto q = drafter({1, 2, 3, 4, 9, 1, 2, 3, 4, 7, 1, 2, 3}, 2).propose({4});
        check(q.size() == 2 && q[0].id == 7 && q[1].id == 9 && q[0].p == 0.5, "drafter: longest match, most recent first on ties");
    }
    {  // only a 3-token suffix matches
        auto q = drafter({5, 1, 2, 3, 8, 0, 1, 2}, 3).propose({3});
        check(q.size() == 1 && q[0].id == 8 && q[0].p == 1.0, "drafter: falls back to shorter suffixes down to nmin");
    }
    check(drafter({5, 6, 7}, 1).propose({8}).empty(), "drafter: no match -> no draft");
    check(drafter({1, 2, 3, 1, 2}, 0).propose({3}).empty(), "drafter: nmin 0 -> off");
    check(drafter({1, 2, 3, 1}, 3).propose({2}).empty(), "drafter: suffix shorter than nmin -> no draft");
    {  // 7 followed by 1, 1, 2, 7 -> q = {1: .5, 7: .25 (more recent), 2: .25}
        auto q = drafter({7, 1, 7, 1, 7, 2, 7}, 1, 1).propose({7});
        check(q.size() == 3 && q[0].id == 1 && q[0].p == 0.5 && q[1].id == 7 && q[2].id == 2, "drafter: counts as q");
    }
    {
        NgramDrafter d = drafter({1, 2, 3}, 1);
        d.propose({1});
        check(d.h.size() == 3, "drafter: propose leaves the history unchanged");
    }
    std::mt19937_64 crng(7);
    {  // chain: "def f ( x ) :" seen before; after "def f" the drafts continue the earlier occurrence, then stop
        NgramDrafter d = drafter({10, 11, 12, 13, 14, 15, 16, 90, 91, 10}, 2);
        std::vector<int> ds; std::vector<std::vector<TokP>> qs;
        d.chain(11, 8, true, crng, ds, qs);
        check(ds == std::vector<int>({12, 13, 14, 15, 16, 90, 91, 10}) && qs.size() == 8 && qs[3].size() == 1 && qs[3][0].p == 1.0,
              "drafter: greedy chain follows the earlier occurrence, q one-hot");
        d.chain(11, 3, true, crng, ds, qs);
        check(ds.size() == 3, "drafter: chain stops at k");
        d.chain(55, 4, true, crng, ds, qs);
        check(ds.empty(), "drafter: chain without a match is empty (MTP head draft instead)");
        NgramDrafter e = drafter({1, 2, 3, 4, 1, 2, 3}, 2);
        e.chain(4, 8, true, crng, ds, qs);
        check(ds.size() == 8 && ds[0] == 1 && ds[3] == 4 && ds[4] == 1, "drafter: chain can run into its own drafts (repeats)");
        NgramDrafter f = drafter({1, 2, 9, 5, 1, 2, 7, 6, 1}, 2);
        f.chain(2, 4, false, crng, ds, qs);
        check(!ds.empty() && (ds[0] == 9 || ds[0] == 7) && qs[0].size() == 2, "drafter: sampled chain keeps q (two continuations)");
    }
    {  // adaptive length
        DraftLen L; L.kmax = 8;
        const int k0 = L.k();
        for (int i = 0; i < 20; i++) L.update(L.k(), L.k());
        const int khi = L.k();
        for (int i = 0; i < 20; i++) L.update(L.k(), 0);
        char msg[128]; snprintf(msg, sizeof msg, "draft length: start %d, after full accepts %d, after rejects %d", k0, khi, L.k());
        check(k0 == 4 && khi == 8 && L.k() == 1, msg);
    }

    // p/q: whatever q is, the token out of a (draft ~ q, spec_accept) step is distributed as p
    const std::vector<TokP> p = {{10, 0.4}, {11, 0.3}, {12, 0.2}, {13, 0.1}};
    const std::vector<std::pair<const char*, std::vector<TokP>>> qs = {
        {"q one-hot on p's top (greedy draft)", {{10, 1.0}}},
        {"q one-hot on an unlikely token", {{13, 1.0}}},
        {"q one-hot outside p's support", {{99, 1.0}}},
        {"q == p", p},
        {"q skewed, partly outside p", {{11, 0.5}, {13, 0.3}, {99, 0.2}}},
    };
    std::mt19937_64 rng(1234);
    const int N = 400000;
    for (auto& [name, q] : qs) {
        std::vector<double> hist(4, 0.0);
        int acc = 0, other = 0;
        for (int i = 0; i < N; i++) {
            const int d = sample_from(q, rng);
            auto [a, t] = spec_accept(p, q, d, rng);
            acc += a;
            if (a && t != d) other++;
            if (t >= 10 && t <= 13) hist[t - 10] += 1.0 / N; else other++;
        }
        double tv = 0, want_acc = 0;
        for (int i = 0; i < 4; i++) tv += 0.5 * std::fabs(hist[i] - p[i].p);
        for (const TokP& e : p) want_acc += std::min(e.p, prob_of(q, e.id));
        char msg[256];
        snprintf(msg, sizeof msg, "p/q, %s: TV(out, p) %.4f, accept %.4f (theory %.4f)", name, tv, (double)acc / N, want_acc);
        check(other == 0 && tv < 0.004 && std::fabs((double)acc / N - want_acc) < 0.004, msg);
    }
    {  // a 3-draft round over a Markov target p(b | a): the first two tokens kept are distributed as p(a | y) p(b | a),
       // whatever the draft chain (here a fixed, partly wrong one, and a sampled one)
        auto P = [](int prev) {
            std::vector<TokP> v;
            for (int t = 0; t < 4; t++) v.push_back({t, (1.0 + ((t + prev) % 4)) / 10.0});  // 0.1 .. 0.4, shifted by prev
            return v;
        };
        const int y = 2;
        const std::vector<TokP> qd = {{0, 0.5}, {1, 0.3}, {3, 0.2}};
        for (int mode = 0; mode < 2; mode++) {
            double hist[4][4] = {}; long long kept = 0;
            for (int n = 0; n < N; n++) {
                std::vector<int> d; std::vector<std::vector<TokP>> qs;
                for (int i = 0; i < 3; i++) {
                    const int t = mode ? sample_from(qd, rng) : (i == 1 ? 3 : 1);
                    d.push_back(t); qs.push_back(mode ? qd : std::vector<TokP>{{t, 1.0}});
                }
                // the target at draft i's position, given the kept tokens before it (the drafts while accepted)
                std::vector<int> out = spec_verify(d, qs, [&](int i) { return P(i ? d[i - 1] : y); }, rng);
                kept += out.size();
                if (out.size() >= 2) hist[out[0]][out[1]] += 1.0;
                else {  // one token: the second comes from the next round, i.e. from p(. | out[0])
                    hist[out[0]][sample_from(P(out[0]), rng)] += 1.0;
                }
            }
            double tv = 0;
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) tv += 0.5 * std::fabs(hist[a][b] / N - P(y)[a].p * P(a)[b].p);
            char msg[160];
            snprintf(msg, sizeof msg, "3-draft round, %s drafts: TV(first two tokens, target) %.4f, %.2f tokens per round",
                     mode ? "sampled" : "fixed", tv, (double)kept / N);
            check(tv < 0.006, msg);
        }
    }
    printf(fails ? "test_spec: %d FAILED\n" : "test_spec: ALL PASS\n", fails);
    return fails != 0;
}
