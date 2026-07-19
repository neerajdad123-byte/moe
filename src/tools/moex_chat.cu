// MoEx: interactive terminal chat.
//
// Caching policy: plain LRU, no hard pinning. This is not a default of
// convenience -- it's the empirical winner from build/bench50/policy_sim_report.html,
// which replayed 37 candidate formulas (frequency pins, co-occurrence/lift/
// Jaccard clusters, EMA, LFU) against the real 50-prompt route trace at this
// same capacity and found every pinning variant scored BELOW plain LRU with
// no protection at all. dm.pin() is never called here.
//
// Warm start: before the first turn, pre-loads the experts that mattered
// most across the historical python50 run (build/bench50/summary.json's
// final_expert_histogram_layer_major), so the first exchange isn't paying a
// fully cold start. This is a pre-load, NOT a pin -- nothing here is
// protected from eviction; if the conversation doesn't need it, LRU reclaims
// the slot exactly like anything else.
//
// Multi-turn: one Forward instance (and its KV cache) lives for the whole
// session, so context carries across turns like a real chat.
//
// Streaming: each token's text prints the instant it's generated, plus a
// live tok/s readout that updates in place (ANSI cursor save/restore) after
// every token, so you see speed react in real time instead of only at the
// end of a turn. Rate excludes the FIRST decode step deliberately -- its
// latency includes carryover from the prefill/decode transition and isn't
// representative of steady-state speed; including it drags the average down
// misleadingly on short turns.
//
// Usage: moex_chat [gguf] [capacity=0(auto)] [warm_top_k=6] [max_new_tok=200]
#include <direct.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
static void setup_console() {
  HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  if (GetConsoleMode(hOut, &mode))
    SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
  SetConsoleOutputCP(CP_UTF8);
}
#else
static void setup_console() {}
#endif

#include "cuda/cuda_common.h"
#include "cuda/device_model.h"
#include "cuda/forward.cuh"
#include "gguf/gguf.h"
#include "model/manifest.h"
#include "model/tokenizer.h"

using namespace moex;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Crude scan for one known field in our own summary.json (no JSON lib needed
// for a single fixed-shape array we generated ourselves).
static std::vector<long> load_histogram(const char* path, uint32_t n_layers,
                                        uint32_t n_experts) {
  std::vector<long> hist((size_t)n_layers * n_experts, 0);
  FILE* f = std::fopen(path, "rb");
  if (!f) return hist;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string buf((size_t)sz, '\0');
  std::fread(&buf[0], 1, (size_t)sz, f);
  std::fclose(f);
  const char* key = "\"final_expert_histogram_layer_major\": [";
  size_t pos = buf.find(key);
  if (pos == std::string::npos) return hist;
  const char* p = buf.c_str() + pos + std::strlen(key);
  size_t i = 0;
  while (i < hist.size()) {
    char* end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p) break;
    hist[i++] = v;
    p = end;
    while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r') ++p;
  }
  return hist;
}

int main(int argc, char** argv) {
  setup_console();
  const char* path =
      argc > 1 ? argv[1] : "C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf";
  uint32_t capacity = argc > 2 ? (uint32_t)std::atoi(argv[2]) : 0;
  // Small on purpose: the offline simulation (policy_sim_report.html) found
  // plain LRU wins because it gives the ENTIRE budget to the current
  // conversation. A large warm-start fights that -- at top_k=20 this reserved
  // 960/1043 slots (92% of the arena) before the first live message, leaving
  // almost no room to breathe and forcing heavy immediate eviction churn.
  // top_k=6 uses <30% of capacity as a light nudge, not a de facto pin.
  uint32_t warm_top_k = argc > 3 ? (uint32_t)std::atoi(argv[3]) : 6;
  int max_new_tok = argc > 4 ? std::atoi(argv[4]) : 200;
  const uint32_t max_ctx = 2048;

  gguf::Model m;
  std::string err;
  if (!m.open(path, &err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  Manifest man;
  if (!man.build(m, &err)) {
    std::fprintf(stderr, "manifest: %s\n", err.c_str());
    return 1;
  }
  Detokenizer detok;
  if (!detok.load(m, &err)) {
    std::fprintf(stderr, "tokenizer(decode): %s\n", err.c_str());
    return 1;
  }
  Encoder enc;
  if (!enc.load(m, &err)) {
    std::fprintf(stderr, "tokenizer(encode): %s\n", err.c_str());
    return 1;
  }
  const ModelConfig& c = man.config();
  const uint8_t* host_base = m.base();

  std::printf("=== MoEx chat === loading model...\n");
  DeviceModel dm;
  if (!dm.upload_static(m, man, host_base, &err)) {
    std::fprintf(stderr, "static upload: %s\n", err.c_str());
    return 1;
  }
  const uint64_t vram_reserve = 900ull << 20;  // KV @ ctx=2048 + workspace + guard
  capacity = dm.init_arena(man, capacity, vram_reserve, &err);
  if (capacity == 0) {
    std::fprintf(stderr, "init_arena: %s\n", err.c_str());
    return 1;
  }
  std::printf("arena: %u slots (%.2f GiB) -- LRU only, no pinning (see file header)\n",
              capacity, capacity * dm.slot_stride() / 1073741824.0);

  // ---- warm start from historical python50 usage (pre-load, not a pin) ----
  {
    auto hist = load_histogram("build/bench50/summary.json", c.n_layers, c.n_experts);
    uint64_t warm_bytes = 0;
    uint32_t warm_n = 0;
    for (uint32_t l = 0; l < c.n_layers; ++l) {
      std::vector<std::pair<long, uint32_t>> ranked;
      for (uint32_t e = 0; e < c.n_experts; ++e) {
        long cnt = hist[(size_t)l * c.n_experts + e];
        if (cnt > 0) ranked.push_back({cnt, e});
      }
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      std::vector<int> ids;
      for (size_t i = 0; i < ranked.size() && i < warm_top_k; ++i)
        ids.push_back((int)ranked[i].second);
      if (!ids.empty()) {
        std::string e2;
        uint64_t up = 0;
        uint32_t nup = 0;
        dm.ensure_bounded(m, man, host_base, l, ids.data(), (uint32_t)ids.size(),
                          &e2, &up, nullptr, &nup);
        warm_bytes += up;
        warm_n += nup;
      }
    }
    if (warm_n > 0)
      std::printf("warm-started %u experts (%.2f MiB) from historical Python usage\n",
                  warm_n, warm_bytes / 1048576.0);
    else
      std::printf("(no warm-start data found at build/bench50/summary.json -- starting cold)\n");
  }

  Forward fwd(dm, c, max_ctx, man);
  uint64_t reactive_up_bytes = 0;
  int reactive_up_calls = 0;
  long turn_hits = 0, turn_miss = 0;
  fwd.ensure_experts = [&](uint32_t layer, const int* ids, uint32_t n) {
    for (uint32_t j = 0; j < n; ++j) {
      int e = ids[j];
      if (e < 0) continue;
      if (dm.expert(layer, (uint32_t)e).resident) ++turn_hits;
      else ++turn_miss;
    }
    std::string e2;
    uint64_t up = 0;
    dm.ensure_bounded(m, man, host_base, layer, ids, n, &e2, &up, nullptr, nullptr);
    reactive_up_bytes += up;
    if (up > 0) ++reactive_up_calls;
  };

  std::vector<int> route_buf((size_t)c.n_layers * c.n_experts_used, -1);
  int pos = 0;
  int eos = detok.eos_id();

  std::printf("\nready. type a message ('exit' to quit).\n");
  std::printf("context is %u tokens (multi-turn; carries across messages in this session)\n\n",
              max_ctx);

  std::string line;
  while (true) {
    std::printf("You: ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line == "exit" || line == "quit") break;
    if (line.empty()) continue;

    std::string enc_err;
    std::vector<int> ids = enc.encode(line, &enc_err);
    if (ids.empty()) {
      std::printf("(couldn't tokenize that -- try again)\n\n");
      continue;
    }
    if (pos + (int)ids.size() + max_new_tok >= (int)max_ctx) {
      std::printf("(context window full for this session -- restart to continue)\n\n");
      continue;
    }

    turn_hits = 0;
    turn_miss = 0;
    uint64_t up_before = reactive_up_bytes;

    auto t0 = Clock::now();
    int tok = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      tok = fwd.step(ids[i], pos, &route_buf, nullptr);
      ++pos;
    }
    double prefill_ms = ms(t0, Clock::now());

    // Streaming decode: print each token's text the instant it's produced,
    // and keep a live tok/s readout updating in place below it. The rate
    // deliberately excludes the FIRST decode step (per_tok_ms[0]) -- its
    // latency carries transition cost from prefill and isn't representative
    // of steady-state speed; folding it in understates the real rate,
    // especially on short replies.
    std::printf("Bot: ");
    std::fflush(stdout);

    std::vector<double> per_tok_ms;  // one entry per fwd.step() decode call
    int n_gen = 0;
    bool printed_live_line = false;
    auto t1 = Clock::now();
    for (int g = 0; g < max_new_tok; ++g) {
      if (tok == eos) break;
      std::string piece = detok.decode_id(tok);
      std::fwrite(piece.data(), 1, piece.size(), stdout);

      auto ta = Clock::now();
      tok = fwd.step(tok, pos, &route_buf, nullptr);
      ++pos;
      auto tb = Clock::now();
      per_tok_ms.push_back(ms(ta, tb));
      ++n_gen;

      if (per_tok_ms.size() >= 2) {
        double sum_ms = 0;
        for (size_t k = 1; k < per_tok_ms.size(); ++k) sum_ms += per_tok_ms[k];
        double live_tok_s = (double)(per_tok_ms.size() - 1) / (sum_ms / 1000.0);
        std::printf("\x1b[s\n\x1b[2K  [%d tok so far, %.2f tok/s (first tok excluded)]\x1b[u",
                    n_gen, live_tok_s);
        printed_live_line = true;
      }
      std::fflush(stdout);
    }
    double decode_ms = ms(t1, Clock::now());
    double sum_ms_excl_first = 0;
    for (size_t k = 1; k < per_tok_ms.size(); ++k) sum_ms_excl_first += per_tok_ms[k];
    double tok_s = per_tok_ms.size() >= 2
                       ? (double)(per_tok_ms.size() - 1) / (sum_ms_excl_first / 1000.0)
                       : 0.0;

    long sel = turn_hits + turn_miss;
    if (printed_live_line) {
      // Overwrite the last live line with the complete final stats (no
      // restore this time -- settle here instead of jumping back).
      std::printf(
          "\x1b[s\n\x1b[2K  [%d tok in %.0fms -> %.2f tok/s (first tok excluded) | "
          "prefill %.0fms | hit %ld/%ld (%.1f%%) | +%.2f MiB H2D]\n\n",
          n_gen, decode_ms, tok_s, prefill_ms, turn_hits, sel,
          sel ? 100.0 * turn_hits / sel : 0.0,
          (reactive_up_bytes - up_before) / 1048576.0);
    } else {
      std::printf(
          "\n  [%d tok in %.0fms -> %.2f tok/s (too short to exclude first tok) | "
          "prefill %.0fms | hit %ld/%ld (%.1f%%) | +%.2f MiB H2D]\n\n",
          n_gen, decode_ms, n_gen > 0 ? n_gen / (decode_ms / 1000.0) : 0.0, prefill_ms,
          turn_hits, sel, sel ? 100.0 * turn_hits / sel : 0.0,
          (reactive_up_bytes - up_before) / 1048576.0);
    }
  }

  std::printf("session end. total reactive H2D: %.2f MiB in %d calls, arena %u/%u resident\n",
              reactive_up_bytes / 1048576.0, reactive_up_calls, dm.resident_expert_count(),
              capacity);
  return 0;
}
