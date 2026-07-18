// MoEx: multi-prompt route-trace + eviction benchmark (Step 2 / hit-rate).
//
// Runs up to 50 embedded code-generation prompts through a bounded, evicting
// expert arena (DeviceModel::ensure_bounded), fresh KV per prompt (nothing
// persists across prompts), with a periodic "pin the cumulative top-K
// experts per layer" refresh every `pin_every` prompts. Dumps every piece of
// per-prompt/per-token/per-layer data the profiler already has plus route
// traces, so the output becomes the input to an offline cluster-mining +
// cache-policy simulator (not part of this tool).
//
// Usage: moex_bench50 [gguf] [num_prompts=10] [decode_tokens=64]
//                      [pin_every=5] [pin_top_k=17] [capacity=0(auto)]
//                      [corpus=mixed50|python10] [passes=1] [profile=0]
//
// capacity=0 auto-sizes the slot-pool arena from live free VRAM (avoids WDDM
// oversubscription spill). profile=0 is the clean headline mode (no CUDA
// events/syncs from the profiler; hit/miss still tracked via upload counts);
// profile=1 attaches StepProf and additionally writes per_token.csv.
//
// `passes` repeats the same `num_prompts`-prompt sequence `passes` times in
// one process (same DeviceModel/arena/pin state throughout — nothing is
// reset between passes). Pass 0 is the cold-start reactive run; pass 1+
// starts with whatever pass 0 left pinned/resident, so pass-vs-pass
// hit-rate/decode-speed deltas answer "does a warm-started pin set actually
// help" directly, without needing cross-process residency persistence.
#include <direct.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "cuda/cuda_common.h"
#include "cuda/device_model.h"
#include "cuda/forward.cuh"
#include "cuda/profiler.cuh"
#include "gguf/gguf.h"
#include "model/manifest.h"
#include "model/tokenizer.h"

using namespace moex;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---- 50 code-only prompts, 8 languages x varied task shapes ---------------
static const char* kPrompts[50] = {
    // Python (7)
    "Write only Python code for a function that sorts a list using quicksort. No comments, no explanation, code only.",
    "Write only Python code for a function that reverses a singly linked list. No comments, no explanation, code only.",
    "Write only Python code for a function that checks whether a string is a palindrome. No comments, no explanation, code only.",
    "Write only Python code for a function that computes the nth Fibonacci number iteratively. No comments, no explanation, code only.",
    "Write only Python code for a stack class implemented using a list. No comments, no explanation, code only.",
    "Write only Python code for a function that performs breadth first search on a graph given as an adjacency list. No comments, no explanation, code only.",
    "Write only Python code for a simple LRU cache class with get and put methods. No comments, no explanation, code only.",
    // C++ (7)
    "Write only C++ code for a function that performs binary search on a sorted array. No comments, no explanation, code only.",
    "Write only C++ code for a function that implements merge sort on a vector of integers. No comments, no explanation, code only.",
    "Write only C++ code for a function that implements bubble sort on an array. No comments, no explanation, code only.",
    "Write only C++ code for a function that detects a cycle in a singly linked list. No comments, no explanation, code only.",
    "Write only C++ code for a min-heap class with push and pop operations. No comments, no explanation, code only.",
    "Write only C++ code for a function that implements Dijkstra's shortest path algorithm on a weighted graph. No comments, no explanation, code only.",
    "Write only C++ code for a function that computes the edit distance between two strings. No comments, no explanation, code only.",
    // JavaScript (6)
    "Write only JavaScript code for a function that computes factorial recursively. No comments, no explanation, code only.",
    "Write only JavaScript code for a function that checks whether two strings are anagrams. No comments, no explanation, code only.",
    "Write only JavaScript code for a function that flattens a nested array. No comments, no explanation, code only.",
    "Write only JavaScript code for a debounce function that wraps another function. No comments, no explanation, code only.",
    "Write only JavaScript code for a function that deep clones a plain object. No comments, no explanation, code only.",
    "Write only JavaScript code for a function that validates balanced parentheses in a string. No comments, no explanation, code only.",
    // Java (6)
    "Write only Java code for a queue class implemented using two stacks. No comments, no explanation, code only.",
    "Write only Java code for a function that performs inorder traversal of a binary tree. No comments, no explanation, code only.",
    "Write only Java code for a function that counts the number of set bits in an integer. No comments, no explanation, code only.",
    "Write only Java code for a function that implements selection sort on an array. No comments, no explanation, code only.",
    "Write only Java code for a function that rotates an array by k positions. No comments, no explanation, code only.",
    "Write only Java code for a doubly linked list class with insert and delete methods. No comments, no explanation, code only.",
    // Go (6)
    "Write only Go code for a function that computes the greatest common divisor of two integers. No comments, no explanation, code only.",
    "Write only Go code for a function that finds the longest common prefix among a slice of strings. No comments, no explanation, code only.",
    "Write only Go code for a function that performs topological sort on a directed acyclic graph. No comments, no explanation, code only.",
    "Write only Go code for a circular buffer struct with push and pop methods. No comments, no explanation, code only.",
    "Write only Go code for a function that computes the sum of digits of an integer. No comments, no explanation, code only.",
    "Write only Go code for a function that finds the missing number in a slice containing n distinct numbers from 0 to n. No comments, no explanation, code only.",
    // Rust (6)
    "Write only Rust code for a function that computes power of a number using fast exponentiation. No comments, no explanation, code only.",
    "Write only Rust code for a trie struct with insert and search methods. No comments, no explanation, code only.",
    "Write only Rust code for a function that finds the kth largest element in an unsorted array. No comments, no explanation, code only.",
    "Write only Rust code for a function that generates all permutations of a string. No comments, no explanation, code only.",
    "Write only Rust code for a function that converts a decimal integer to its binary string representation. No comments, no explanation, code only.",
    "Write only Rust code for a basic calculator function that evaluates an expression with plus minus times and divide. No comments, no explanation, code only.",
    // SQL (6)
    "Write only SQL code for a query that selects all rows from a table named users where age is greater than 30. No comments, no explanation, code only.",
    "Write only SQL code for a query that joins a table named orders with a table named customers on customer_id. No comments, no explanation, code only.",
    "Write only SQL code for a query that counts rows grouped by department from a table named employees. No comments, no explanation, code only.",
    "Write only SQL code for a query that finds the second highest salary from a table named employees. No comments, no explanation, code only.",
    "Write only SQL code for a query that deletes duplicate rows from a table named events keeping only the lowest id. No comments, no explanation, code only.",
    "Write only SQL code for a query that updates the status column to inactive where last_login is older than one year in a table named users. No comments, no explanation, code only.",
    // Bash (6)
    "Write only Bash code for a script that lists all .txt files in the current directory. No comments, no explanation, code only.",
    "Write only Bash code for a script that counts the number of lines in a file given as an argument. No comments, no explanation, code only.",
    "Write only Bash code for a script that finds and replaces a string in all files in a directory. No comments, no explanation, code only.",
    "Write only Bash code for a script that backs up a directory into a tar.gz archive. No comments, no explanation, code only.",
    "Write only Bash code for a script that checks disk usage and prints a warning if it is over 90 percent. No comments, no explanation, code only.",
    "Write only Bash code for a script that loops through its arguments and prints each one reversed. No comments, no explanation, code only.",
};

// All-Python, 10-prompt corpus for the same-domain repeat-pass diagnostic.
// Phrased to elicit a single clean code block (no prose) for easy eyeballing.
static const char* kPromptsPython10[10] = {
    "Respond with only a single Python markdown code block containing a function that sorts a list using quicksort. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that reverses a singly linked list. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that checks whether a string is a palindrome. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that computes the nth Fibonacci number iteratively. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a stack class implemented using a list. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that performs breadth first search on a graph given as an adjacency list. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a simple LRU cache class with get and put methods. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that performs binary search on a sorted list. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that implements merge sort on a list of integers. Do not include any text before or after the code block. Do not include comments inside the code.",
    "Respond with only a single Python markdown code block containing a function that implements bubble sort on a list. Do not include any text before or after the code block. Do not include comments inside the code.",
};

static std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o.push_back((char)c);
        }
    }
  }
  return o;
}

struct PromptRecord {
  int idx = 0;
  std::string prompt_text;
  int n_prompt_tokens = 0;
  int n_decode_tokens = 0;
  double ttft_ms = 0;
  double prefill_tok_s = 0;
  double decode_tok_s = 0;
  std::vector<double> per_tok_ms;
  long hits = 0, miss = 0;
  std::string output_text;
  size_t trace_begin = 0, trace_end = 0;  // slice into prof.trace / prof.tok_ms
  int pass_idx = 0;  // 0 = first time through the corpus, 1 = repeat, ...
};

int main(int argc, char** argv) {
  const char* path =
      argc > 1 ? argv[1] : "C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf";
  int num_prompts = argc > 2 ? std::atoi(argv[2]) : 10;
  int decode_tokens = argc > 3 ? std::atoi(argv[3]) : 64;
  int pin_every = argc > 4 ? std::atoi(argv[4]) : 5;
  uint32_t pin_top_k = argc > 5 ? (uint32_t)std::atoi(argv[5]) : 17;
  uint32_t capacity = argc > 6 ? (uint32_t)std::atoi(argv[6]) : 0;  // 0 = auto
  std::string corpus = argc > 7 ? argv[7] : "mixed50";
  int passes = argc > 8 ? std::atoi(argv[8]) : 1;
  int profile = argc > 9 ? std::atoi(argv[9]) : 0;  // 0 = clean headline mode
  if (num_prompts < 1) num_prompts = 1;
  if (num_prompts > 50) num_prompts = 50;
  if (decode_tokens < 1) decode_tokens = 1;
  if (passes < 1) passes = 1;
  const uint32_t max_ctx = 1024;

  const char* const* corpus_arr = kPrompts;
  int corpus_size = 50;
  if (corpus == "python10") {
    corpus_arr = kPromptsPython10;
    corpus_size = 10;
    if (num_prompts > corpus_size) num_prompts = corpus_size;
  }
  int total_iters = num_prompts * passes;

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

  _mkdir("build");
  _mkdir("build\\bench50");

  size_t free0 = 0, total0 = 0;
  MOEX_CUDA(cudaMemGetInfo(&free0, &total0));
  std::printf("=== MoEx bench50: %d prompts x %d decode tok x %d pass(es), corpus=%s, capacity=%u, pin top-%u/layer every %d ===\n",
              num_prompts, decode_tokens, passes, corpus.c_str(), capacity,
              pin_top_k, pin_every);
  std::printf("model:  %s\n", path);
  std::printf("config: layers=%u experts=%u top_k=%u d_model=%u\n", c.n_layers,
              c.n_experts, c.n_experts_used, c.d_model);
  std::printf("GPU:    %.0f / %.0f MiB free before load\n", free0 / 1048576.0,
              total0 / 1048576.0);

  DeviceModel dm;
  if (!dm.upload_static(m, man, host_base, &err)) {
    std::fprintf(stderr, "static upload: %s\n", err.c_str());
    return 1;
  }
  std::printf("static resident: %s\n", human_bytes(dm.bytes_resident()).c_str());

  // Slot-pool arena. Reserve covers: KV cache for max_ctx (48*2*512*4B*1024
  // = 192 MiB), Forward workspaces/logits (~100 MiB), plus a WDDM/other-app
  // guard band so we never oversubscribe physical VRAM (which Windows
  // "handles" by silently spilling GPU memory to system RAM — a huge hidden
  // slowdown observed in the previous run at 6140/6141 MiB used).
  const uint64_t vram_reserve = 700ull << 20;
  capacity = dm.init_arena(man, capacity, vram_reserve, &err);
  if (capacity == 0) {
    std::fprintf(stderr, "init_arena: %s\n", err.c_str());
    return 1;
  }
  std::printf("expert arena:    %u slots x %.2f MiB (%s), pinned ring 32 x %.2f MiB\n",
              capacity, dm.slot_stride() / 1048576.0,
              human_bytes((uint64_t)capacity * dm.slot_stride()).c_str(),
              dm.slot_stride() / 1048576.0);

  FILE* route_f = std::fopen("build/bench50/route_trace.jsonl", "wb");
  FILE* pins_f = std::fopen("build/bench50/pins_log.jsonl", "wb");
  if (!route_f || !pins_f) {
    std::fprintf(stderr, "failed to open output files under build/bench50/\n");
    return 1;
  }

  StepProf prof;  // attached only when profile=1 (clean headline otherwise).
                  // Never reset across prompts; prompt boundaries recorded
                  // separately for per-prompt latency slicing.

  uint64_t reactive_up_bytes = 0, reactive_evict_bytes = 0;
  int reactive_up_calls = 0;
  uint32_t reactive_n_up = 0;  // experts uploaded reactively == true misses
  long evictions_total = 0;
  uint64_t pin_up_bytes = 0, pin_evict_bytes = 0;
  int pin_up_calls = 0;
  std::vector<std::vector<uint32_t>> current_pins(c.n_layers);
  // Tool-side cumulative usage histogram (layer-major), independent of the
  // profiler so pin refresh works in clean mode too.
  std::vector<long> tool_hist((size_t)c.n_layers * c.n_experts, 0);

  std::vector<PromptRecord> records;
  records.reserve(total_iters);

  for (int p = 0; p < total_iters; ++p) {
    const char* ptext = corpus_arr[p % num_prompts];
    std::string enc_err;
    std::vector<int> ids = enc.encode(ptext, &enc_err);
    if (!enc_err.empty())
      std::fprintf(stderr, "[prompt %d] encode warning: %s\n", p, enc_err.c_str());
    if (ids.empty()) {
      std::fprintf(stderr, "[prompt %d] encoded to 0 tokens, skipping\n", p);
      continue;
    }

    PromptRecord rec;
    rec.idx = p;
    rec.pass_idx = p / num_prompts;
    rec.prompt_text = ptext;
    rec.n_prompt_tokens = (int)ids.size();
    rec.trace_begin = prof.trace.size();

    Forward fwd(dm, c, max_ctx);
    fwd.prof = profile ? &prof : nullptr;
    fwd.ensure_experts = [&](uint32_t layer, const int* rids, uint32_t n) {
      std::string e2;
      uint64_t up = 0, ev = 0;
      uint32_t evn = dm.ensure_bounded(m, man, host_base, layer, rids, n, &e2,
                                       &up, &ev, &reactive_n_up);
      reactive_up_bytes += up;
      reactive_evict_bytes += ev;
      if (up > 0) ++reactive_up_calls;
      evictions_total += evn;
      if (!e2.empty())
        std::fprintf(stderr, "[prompt %d] ensure_bounded: %s\n", p, e2.c_str());
    };

    auto log_route = [&](const char* phase, int pos,
                         const std::vector<int>& route,
                         const std::vector<float>& weight) {
      for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::fprintf(route_f,
                     "{\"prompt\":%d,\"phase\":\"%s\",\"pos\":%d,\"layer\":%u,\"experts\":[",
                     p, phase, pos, l);
        for (uint32_t j = 0; j < c.n_experts_used; ++j) {
          std::fprintf(route_f, "%s%d", j ? "," : "",
                       route[(size_t)l * c.n_experts_used + j]);
        }
        std::fprintf(route_f, "],\"weights\":[");
        for (uint32_t j = 0; j < c.n_experts_used; ++j) {
          std::fprintf(route_f, "%s%.6f", j ? "," : "",
                       weight[(size_t)l * c.n_experts_used + j]);
        }
        std::fprintf(route_f, "]}\n");
      }
    };

    std::vector<int> route_buf((size_t)c.n_layers * c.n_experts_used, -1);
    std::vector<float> weight_buf((size_t)c.n_layers * c.n_experts_used, 0.f);

    auto bump_hist = [&]() {
      for (uint32_t l = 0; l < c.n_layers; ++l)
        for (uint32_t j = 0; j < c.n_experts_used; ++j) {
          int e = route_buf[(size_t)l * c.n_experts_used + j];
          if (e >= 0) tool_hist[(size_t)l * c.n_experts + e] += 1;
        }
    };

    int pos = 0;
    int tok = ids[0];
    uint32_t up_before = reactive_n_up;
    long steps = 0;
    auto t0 = Clock::now();
    for (size_t i = 0; i < ids.size(); ++i) {
      tok = fwd.step(ids[i], pos, &route_buf, &weight_buf);
      log_route("prefill", pos, route_buf, weight_buf);
      bump_hist();
      ++steps;
      ++pos;
    }
    rec.ttft_ms = ms(t0, Clock::now());
    rec.prefill_tok_s = rec.n_prompt_tokens / (rec.ttft_ms / 1000.0);

    std::vector<int> generated;
    auto t1 = Clock::now();
    for (int g = 0; g < decode_tokens; ++g) {
      generated.push_back(tok);
      if (tok == detok.eos_id()) break;
      auto ta = Clock::now();
      tok = fwd.step(tok, pos, &route_buf, &weight_buf);
      rec.per_tok_ms.push_back(ms(ta, Clock::now()));
      log_route("decode", pos, route_buf, weight_buf);
      bump_hist();
      ++steps;
      ++pos;
    }
    double decode_ms = ms(t1, Clock::now());
    rec.n_decode_tokens = (int)rec.per_tok_ms.size();
    if (rec.n_decode_tokens > 0)
      rec.decode_tok_s = rec.n_decode_tokens / (decode_ms / 1000.0);
    rec.output_text = detok.decode(generated);
    // Miss = a routed expert that had to be uploaded reactively; hit = the
    // rest. Works in both clean and profiled mode (no profiler dependency).
    rec.miss = (long)(reactive_n_up - up_before);
    rec.hits = steps * (long)c.n_layers * c.n_experts_used - rec.miss;
    rec.trace_end = prof.trace.size();

    std::printf("\n--- pass %d, prompt %d/%d (iter %d/%d) ---\n", rec.pass_idx,
                (p % num_prompts) + 1, num_prompts, p + 1, total_iters);
    std::printf("  in:  %s\n", ptext);
    std::printf("  out: %s\n", rec.output_text.c_str());
    std::printf("  ttft %.1fms  prefill %.2f tok/s  decode %.2f tok/s  hit %ld/%ld (%.1f%%)\n",
                rec.ttft_ms, rec.prefill_tok_s, rec.decode_tok_s, rec.hits,
                rec.hits + rec.miss,
                (rec.hits + rec.miss) ? 100.0 * rec.hits / (rec.hits + rec.miss) : 0.0);

    records.push_back(std::move(rec));

    // --- periodic pin refresh: every pin_every prompts, replace the pinned
    // set with the current cumulative top-pin_top_k experts per layer. ---
    if ((p + 1) % pin_every == 0) {
      uint64_t before_up = pin_up_bytes, before_ev = pin_evict_bytes;
      for (uint32_t l = 0; l < c.n_layers; ++l) {
        for (uint32_t e : current_pins[l]) dm.pin(l, e, false);
        current_pins[l].clear();

        std::vector<std::pair<long, uint32_t>> ranked;
        ranked.reserve(c.n_experts);
        for (uint32_t e = 0; e < c.n_experts; ++e) {
          long cnt = tool_hist[(size_t)l * c.n_experts + e];
          if (cnt > 0) ranked.push_back({cnt, e});
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        uint32_t take = (uint32_t)std::min((size_t)pin_top_k, ranked.size());

        std::vector<int> pin_ids;
        pin_ids.reserve(take);
        for (uint32_t i = 0; i < take; ++i) {
          uint32_t e = ranked[i].second;
          dm.pin(l, e, true);
          current_pins[l].push_back(e);
          pin_ids.push_back((int)e);
        }
        if (!pin_ids.empty()) {
          std::string e2;
          uint64_t up = 0, ev = 0;
          dm.ensure_bounded(m, man, host_base, l, pin_ids.data(),
                            (uint32_t)pin_ids.size(), &e2, &up, &ev);
          pin_up_bytes += up;
          pin_evict_bytes += ev;
          if (up > 0) ++pin_up_calls;
        }
      }
      std::fprintf(pins_f, "{\"after_prompt\":%d,\"pin_top_k\":%u,\"upload_bytes\":%llu,\"evict_bytes\":%llu,\"layer_pins\":{",
                  p + 1, pin_top_k, (unsigned long long)(pin_up_bytes - before_up),
                  (unsigned long long)(pin_evict_bytes - before_ev));
      for (uint32_t l = 0; l < c.n_layers; ++l) {
        std::fprintf(pins_f, "%s\"%u\":[", l ? "," : "", l);
        for (size_t i = 0; i < current_pins[l].size(); ++i)
          std::fprintf(pins_f, "%s%u", i ? "," : "", current_pins[l][i]);
        std::fprintf(pins_f, "]");
      }
      std::fprintf(pins_f, "}}\n");
      std::fflush(pins_f);
      std::printf("  [pin refresh after prompt %d: top-%u/layer, +%.2f MiB uploaded, -%.2f MiB evicted]\n",
                  p + 1, pin_top_k, (pin_up_bytes - before_up) / 1048576.0,
                  (pin_evict_bytes - before_ev) / 1048576.0);
    }
  }
  std::fclose(route_f);
  std::fclose(pins_f);

  // ---- per_token.csv, sliced from prof.trace by recorded prompt bounds ----
  // (profiled mode only — clean mode records no TokenTrace rows)
  if (profile) {
  FILE* csv_f = std::fopen("build/bench50/per_token.csv", "wb");
  std::fprintf(csv_f,
              "prompt_idx,pos,input_token,wall_ms,embed_gpu_ms,attention_gpu_ms,router_gpu_ms,experts_gpu_ms,final_gpu_ms,h2d_bytes,d2h_bytes,d2d_bytes,vram_used_bytes\n");
  for (const auto& rec : records) {
    for (size_t i = rec.trace_begin; i < rec.trace_end; ++i) {
      const TokenTrace& t = prof.trace[i];
      std::fprintf(csv_f, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu\n",
                  rec.idx, t.pos, t.input_token, t.wall_ms, t.embed_gpu_ms,
                  t.attention_gpu_ms, t.router_gpu_ms, t.experts_gpu_ms,
                  t.final_gpu_ms, (unsigned long long)t.h2d_bytes,
                  (unsigned long long)t.d2h_bytes, (unsigned long long)t.d2d_bytes,
                  (unsigned long long)t.vram_used);
    }
  }
  std::fclose(csv_f);
  }

  // ---- summary.json ---------------------------------------------------
  FILE* sum_f = std::fopen("build/bench50/summary.json", "wb");
  std::fprintf(sum_f, "{\n \"prompts\": [\n");
  for (size_t pi = 0; pi < records.size(); ++pi) {
    const auto& r = records[pi];
    std::vector<double> lat = r.per_tok_ms;
    std::fprintf(sum_f,
                "  {\"idx\":%d,\"prompt\":\"%s\",\"n_prompt_tokens\":%d,\"n_decode_tokens\":%d,"
                "\"ttft_ms\":%.3f,\"prefill_tok_s\":%.3f,\"decode_tok_s\":%.3f,"
                "\"hits\":%ld,\"miss\":%ld,\"hit_rate_pct\":%.3f,"
                "\"latency_ms\":{\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,\"p95\":%.3f,\"p99\":%.3f,\"max\":%.3f,\"mean\":%.3f},"
                "\"output\":\"%s\"}%s\n",
                r.idx, json_escape(r.prompt_text).c_str(), r.n_prompt_tokens,
                r.n_decode_tokens, r.ttft_ms, r.prefill_tok_s, r.decode_tok_s,
                r.hits, r.miss,
                (r.hits + r.miss) ? 100.0 * r.hits / (r.hits + r.miss) : 0.0,
                pct(lat, 0), pct(lat, 50), pct(lat, 90), pct(lat, 95),
                pct(lat, 99), pct(lat, 100),
                lat.empty() ? 0.0
                            : std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size(),
                json_escape(r.output_text).c_str(),
                pi + 1 < records.size() ? "," : "");
  }
  std::fprintf(sum_f, " ],\n");

  long total_hits = 0, total_miss = 0;
  for (const auto& r : records) {
    total_hits += r.hits;
    total_miss += r.miss;
  }
  size_t fb, tb;
  cudaMemGetInfo(&fb, &tb);
  std::fprintf(sum_f,
              " \"run\": {\"num_prompts\":%d,\"passes\":%d,\"corpus\":\"%s\",\"decode_tokens_per_prompt\":%d,"
              "\"pin_every\":%d,\"pin_top_k\":%u,\"capacity_experts\":%u,"
              "\"overall_hits\":%ld,\"overall_miss\":%ld,\"overall_hit_rate_pct\":%.3f,"
              "\"evictions\":%ld,\"resident_experts_at_end\":%u,"
              "\"reactive_upload_bytes\":%llu,\"reactive_upload_calls\":%d,\"reactive_evict_bytes\":%llu,"
              "\"pin_upload_bytes\":%llu,\"pin_upload_calls\":%d,\"pin_evict_bytes\":%llu,"
              "\"dequant_bytes_in\":%llu,\"dequant_elems_out\":%llu,"
              "\"h2d_bytes_total\":%llu,\"h2d_calls_total\":%d,"
              "\"d2h_bytes_total\":%llu,\"d2h_calls_total\":%d,"
              "\"d2d_bytes_total\":%llu,\"d2d_calls_total\":%d,"
              "\"kernel_launches_total\":%ld,"
              "\"vram_peak_used_bytes\":%llu,\"vram_free_at_end_MiB\":%.0f,\"vram_total_MiB\":%.0f,"
              "\"kernel_launch_counts\": {",
              num_prompts, passes, corpus.c_str(), decode_tokens, pin_every, pin_top_k, capacity,
              total_hits, total_miss,
              (total_hits + total_miss) ? 100.0 * total_hits / (total_hits + total_miss) : 0.0,
              evictions_total, dm.resident_expert_count(),
              (unsigned long long)reactive_up_bytes, reactive_up_calls,
              (unsigned long long)reactive_evict_bytes,
              (unsigned long long)pin_up_bytes, pin_up_calls,
              (unsigned long long)pin_evict_bytes,
              (unsigned long long)prof.dequant_bytes_in,
              (unsigned long long)prof.dequant_elems_out,
              (unsigned long long)prof.h2d_bytes, prof.h2d_calls,
              (unsigned long long)prof.d2h_bytes, prof.d2h_calls,
              (unsigned long long)prof.d2d_bytes, prof.d2d_calls,
              prof.launches.total, (unsigned long long)prof.vram_peak_used,
              fb / 1048576.0, tb / 1048576.0);
  bool first_k = true;
  for (const auto& kv : prof.launches.calls) {
    std::fprintf(sum_f, "%s\"%s\":%ld", first_k ? "" : ",", kv.first.c_str(), kv.second);
    first_k = false;
  }
  std::fprintf(sum_f, "},\n \"final_expert_histogram_layer_major\": [");
  for (size_t i = 0; i < tool_hist.size(); ++i)
    std::fprintf(sum_f, "%s%ld", i ? "," : "", tool_hist[i]);
  std::fprintf(sum_f, "]\n }\n}\n");
  std::fclose(sum_f);

  std::printf("\n=== SUMMARY (%s mode) ===\n", profile ? "profiled" : "clean headline");
  std::printf("prompts run:        %zu (%d prompts x %d passes)\n", records.size(), num_prompts, passes);
  std::printf("overall hit rate:   %ld / %ld (%.2f%%)\n", total_hits, total_miss + total_hits,
              (total_hits + total_miss) ? 100.0 * total_hits / (total_hits + total_miss) : 0.0);
  std::printf("evictions:          %ld\n", evictions_total);
  std::printf("resident at end:    %u / %u capacity\n", dm.resident_expert_count(), capacity);
  std::printf("reactive H2D:       %.2f MiB in %d calls\n", reactive_up_bytes / 1048576.0, reactive_up_calls);
  std::printf("pin-refresh upload: %.2f MiB in %d calls\n", pin_up_bytes / 1048576.0, pin_up_calls);
  if (profile) {
    std::printf("dequant bytes:      %.3f GiB\n", prof.dequant_bytes_in / 1073741824.0);
    std::printf("kernel launches:    %ld total\n", prof.launches.total);
    std::printf("VRAM peak used:     %.1f MiB\n", prof.vram_peak_used / 1048576.0);
  }
  std::printf("VRAM now used:      %.0f / %.0f MiB (%.0f free — no oversubscription if free > 0)\n",
              (tb - fb) / 1048576.0, tb / 1048576.0, fb / 1048576.0);

  // ---- first-N vs last-N, and pass-vs-pass, aggregate comparison --------
  auto agg = [&](size_t lo, size_t hi) {
    long h = 0, ms_ = 0;
    double sum_decode = 0;
    int n = 0;
    for (size_t i = lo; i < hi && i < records.size(); ++i) {
      h += records[i].hits;
      ms_ += records[i].miss;
      sum_decode += records[i].decode_tok_s;
      ++n;
    }
    double hr = (h + ms_) ? 100.0 * h / (h + ms_) : 0.0;
    double dts = n ? sum_decode / n : 0.0;
    return std::make_pair(hr, dts);
  };
  std::printf("\n=== FIRST-N vs LAST-N (within the %d-prompt sequence) ===\n", num_prompts);
  size_t half = records.size() / 2;
  auto first_half = agg(0, half);
  auto last_half = agg(records.size() - half, records.size());
  std::printf("first %zu prompts: hit rate %.1f%%  mean decode %.2f tok/s\n",
              half, first_half.first, first_half.second);
  std::printf("last  %zu prompts: hit rate %.1f%%  mean decode %.2f tok/s\n",
              half, last_half.first, last_half.second);
  size_t five = std::min((size_t)5, records.size() / 2);
  if (five > 0) {
    auto first5 = agg(0, five);
    auto last5 = agg(records.size() - five, records.size());
    std::printf("first %zu prompts: hit rate %.1f%%  mean decode %.2f tok/s\n",
                five, first5.first, first5.second);
    std::printf("last  %zu prompts: hit rate %.1f%%  mean decode %.2f tok/s\n",
                five, last5.first, last5.second);
  }
  if (passes > 1) {
    std::printf("\n=== PASS-VS-PASS (does a warm-started pin set from the prior pass help) ===\n");
    for (int ps = 0; ps < passes; ++ps) {
      size_t lo = (size_t)ps * num_prompts, hi = lo + num_prompts;
      auto a = agg(lo, hi);
      std::printf("pass %d (prompts %zu..%zu): hit rate %.1f%%  mean decode %.2f tok/s\n",
                  ps, lo, hi - 1, a.first, a.second);
    }
  }

  std::printf("\nwrote build/bench50/route_trace.jsonl, per_token.csv, pins_log.jsonl, summary.json\n");
  return 0;
}
