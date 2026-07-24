// MoEx: pin experts for "hi", then re-run the same prompt fully from VRAM
// (3B-style: no mid-token H2D). Default: 3 identical "hi" passes after pin.
//
// Usage: moex_generate [gguf] [n_gen] [n_repeats] [gpu_dispatch]
//   n_gen        = tokens to generate after "hi" on each pass (default 4)
//   n_repeats    = how many times to re-send "hi" after pin (default 3)
//   gpu_dispatch = 1 (default) for Phase 1's zero-host-round-trip fast path
//                  (router top-k + expert pointer resolution + dispatch all
//                  stay on GPU), 0 for the original per-layer host path
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
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

struct PassResult {
  double prefill_ms = 0;
  double decode_ms = 0;
  double tok_s = 0;
  int n_gen = 0;
  long hits = 0;
  long miss = 0;
  int h2d_calls = 0;
  uint64_t h2d_bytes = 0;
  std::string text;
  std::vector<double> per_tok_ms;
};

// One full "hi" + n_gen decode tokens. ensure_experts must already be set.
static PassResult run_hi(Forward& fwd, const ModelConfig& c,
                         const Detokenizer& detok,
                         const std::vector<int>& prompt, int n_gen,
                         int* h2d_calls_acc, uint64_t* h2d_bytes_acc) {
  PassResult r;
  std::vector<int> route((size_t)c.n_layers * c.n_experts_used, -1);
  int pos = 0;
  int tok = prompt[0];

  int h2d0 = h2d_calls_acc ? *h2d_calls_acc : 0;
  uint64_t b0 = h2d_bytes_acc ? *h2d_bytes_acc : 0;

  auto t0 = Clock::now();
  for (size_t i = 0; i < prompt.size(); ++i)
    tok = fwd.step(prompt[i], pos++, &route);
  r.prefill_ms = ms(t0, Clock::now());

  std::vector<int> generated;
  auto t1 = Clock::now();
  for (int g = 0; g < n_gen; ++g) {
    generated.push_back(tok);
    if (tok == detok.eos_id()) break;
    auto ta = Clock::now();
    tok = fwd.step(tok, pos++, &route);
    r.per_tok_ms.push_back(ms(ta, Clock::now()));
  }
  r.decode_ms = ms(t1, Clock::now());
  r.n_gen = (int)r.per_tok_ms.size();
  if (r.n_gen > 0) r.tok_s = r.n_gen / (r.decode_ms / 1000.0);
  r.text = detok.decode(generated);

  if (h2d_calls_acc) r.h2d_calls = *h2d_calls_acc - h2d0;
  if (h2d_bytes_acc) r.h2d_bytes = *h2d_bytes_acc - b0;
  return r;
}

int main(int argc, char** argv) {
  const char* path =
      argc > 1 ? argv[1] : "C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf";
  int n_gen = argc > 2 ? std::atoi(argv[2]) : 4;
  int n_repeats = argc > 3 ? std::atoi(argv[3]) : 3;
  bool use_gpu_dispatch = argc > 4 ? (std::atoi(argv[4]) != 0) : true;
  if (n_repeats < 1) n_repeats = 1;
  if (n_gen < 1) n_gen = 1;
  const uint32_t max_ctx = 256;

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
    std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
    return 1;
  }
  const ModelConfig& c = man.config();
  const uint8_t* host_base = m.base();

  size_t free0 = 0, total0 = 0;
  MOEX_CUDA(cudaMemGetInfo(&free0, &total0));
  std::printf("=== MoEx: PIN then \"hi\" x%d (VRAM-only, 3B-style) ===\n",
              n_repeats);
  std::printf("gpu_dispatch: %s (Phase 1 + profile; Phase 0 discovery always\n"
              "  uses the host round-trip path for reactive pin)\n",
              use_gpu_dispatch ? "ON — zero host round trips per layer" : "off");
  std::printf("cuda_graph:   %s\n",
              use_gpu_dispatch ? "ON for Phase 1 warm passes (capture then replay)"
                               : "off");
  std::printf("q8_experts:   ON (global Q8 gate/up/down; pin+decode matched)\n");
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

  // ---- Phase 0: DISCOVER experts for "hi" (+ n_gen) via reactive pin ------
  std::set<std::pair<int, int>> pinned;
  uint64_t h2d_bytes = 0;
  int h2d_calls = 0;
  double h2d_ms = 0;

  auto make_uploader = [&](bool allow_upload) {
    return [&, allow_upload](uint32_t layer, const int* ids, uint32_t n) {
      if (!allow_upload) return;
      std::vector<std::vector<uint32_t>> which(c.n_layers);
      bool any = false;
      for (uint32_t j = 0; j < n; ++j) {
        int e = ids[j];
        if (e < 0) continue;
        if (pinned.insert({(int)layer, e}).second) {
          which[layer].push_back((uint32_t)e);
          any = true;
        }
      }
      if (!any) return;
      std::string e2;
      auto ta = Clock::now();
      uint64_t before = dm.bytes_resident();
      dm.upload_experts(m, man, host_base, which, &e2);
      uint64_t added = dm.bytes_resident() - before;
      h2d_bytes += added;
      h2d_calls += 1;
      h2d_ms += ms(ta, Clock::now());
    };
  };

  // "hi" = Qwen3 token 6023
  std::vector<int> prompt = {6023};

  std::printf("\n=== PHASE 0: DISCOVER (reactive pin on first \"hi\") ===\n");
  {
    Forward disc(dm, c, max_ctx, man);
    disc.ensure_experts = make_uploader(true);
    PassResult cold = run_hi(disc, c, detok, prompt, n_gen, &h2d_calls,
                             &h2d_bytes);
    size_t fb, tb;
    cudaMemGetInfo(&fb, &tb);
    std::printf("prompt: \"hi\"  output: \"%s\"\n", cold.text.c_str());
    std::printf("experts pinned:   %zu  (layer-scoped bundles)\n", pinned.size());
    std::printf("expert H2D:       %.2f MiB in %d uploads, %.1f ms (%.2f GB/s)\n",
                h2d_bytes / 1048576.0, h2d_calls, h2d_ms,
                h2d_ms > 0 ? (h2d_bytes / 1e9) / (h2d_ms / 1000.0) : 0.0);
    std::printf("total resident:   %s\n",
                human_bytes(dm.bytes_resident()).c_str());
    std::printf("VRAM:             %llu / %llu MiB  (%.0f free)\n",
                (unsigned long long)((tb - fb) >> 20),
                (unsigned long long)(tb >> 20), (double)(fb >> 20));
    std::printf("cold prefill:     %.1f ms  | cold decode: %.2f tok/s "
                "(%d tok)\n",
                cold.prefill_ms, cold.tok_s, cold.n_gen);
    std::printf("NOTE: cold pass paid H2D — not the 3B-style number.\n");
  }

  // ---- Phase 1..N: experts already in VRAM; fresh KV each time; no H2D ----
  std::printf("\n=== PHASE 1: EXPERTS PRE-PINNED — re-send \"hi\" %d times ===\n",
              n_repeats);
  std::printf("(fresh KV each pass, live router, H2D disabled — pure VRAM)\n");

  long total_hits = 0, total_miss = 0;
  int total_h2d = 0;

  // Hit/miss probe via ensure_experts (no upload).
  auto make_probe = [&](long* hits, long* miss) {
    return [hits, miss, &dm](uint32_t layer, const int* ids, uint32_t n) {
      for (uint32_t j = 0; j < n; ++j) {
        int e = ids[j];
        if (e < 0) continue;
        if (dm.expert(layer, (uint32_t)e).resident) ++(*hits);
        else ++(*miss);
      }
    };
  };

  std::vector<PassResult> results;
  results.reserve(n_repeats);

  for (int rep = 1; rep <= n_repeats; ++rep) {
    // Brand-new Forward = empty KV, same DeviceModel (experts stay resident).
    Forward fwd(dm, c, max_ctx, man);
    fwd.gpu_dispatch = use_gpu_dispatch;
    fwd.use_cuda_graph = use_gpu_dispatch;
    long hits = 0, miss = 0;
    int h2d_before = h2d_calls;
    uint64_t b_before = h2d_bytes;
    // No upload — only count residency. Unused when gpu_dispatch is on
    // (that path never calls ensure_experts); harmless to leave set.
    fwd.ensure_experts = make_probe(&hits, &miss);

    PassResult pr = run_hi(fwd, c, detok, prompt, n_gen, &h2d_calls, &h2d_bytes);
    if (use_gpu_dispatch) {
      unsigned long long ghits = 0, gmiss = 0;
      fwd.read_hit_miss_counters(&ghits, &gmiss);
      hits = (long)ghits;
      miss = (long)gmiss;
    }
    pr.hits = hits;
    pr.miss = miss;
    pr.h2d_calls = h2d_calls - h2d_before;
    pr.h2d_bytes = h2d_bytes - b_before;
    results.push_back(pr);
    total_hits += hits;
    total_miss += miss;
    total_h2d += pr.h2d_calls;

    double mean_tok = 0;
    for (double x : pr.per_tok_ms) mean_tok += x;
    if (!pr.per_tok_ms.empty()) mean_tok /= pr.per_tok_ms.size();
    long sel = hits + miss;
    double hit_pct = sel ? 100.0 * hits / sel : 0.0;

    std::printf("\n--- pass %d/%d ---\n", rep, n_repeats);
    std::printf("  output:     \"%s\"\n", pr.text.c_str());
    std::printf("  prefill:    %.1f ms   (TTFT / first-id ready)\n",
                pr.prefill_ms);
    std::printf("  decode:     %.2f tok/s  (%d tok in %.1f ms, mean %.1f ms/tok)\n",
                pr.tok_s, pr.n_gen, pr.decode_ms, mean_tok);
    if (!pr.per_tok_ms.empty()) {
      std::printf("  per-tok ms: min %.1f  p50 %.1f  p95 %.1f  max %.1f\n",
                  pct(pr.per_tok_ms, 0), pct(pr.per_tok_ms, 50),
                  pct(pr.per_tok_ms, 95), pct(pr.per_tok_ms, 100));
    }
    std::printf("  expert hit: %ld / %ld  (%.1f%%)  miss=%ld\n", hits, sel,
                hit_pct, miss);
    std::printf("  H2D this pass: %d calls, %.2f MiB  %s\n", pr.h2d_calls,
                pr.h2d_bytes / 1048576.0,
                pr.h2d_calls == 0 ? "(GOOD — pure VRAM)" : "(BAD — still paging)");
  }

  // ---- Summary -------------------------------------------------------------
  size_t fb, tb;
  cudaMemGetInfo(&fb, &tb);
  std::printf("\n=== SUMMARY (pinned \"hi\" x%d) ===\n", n_repeats);
  std::printf("experts pre-pinned:  %zu\n", pinned.size());
  std::printf("MoEx resident:       %s\n",
              human_bytes(dm.bytes_resident()).c_str());
  std::printf("VRAM now:            %llu / %llu MiB\n",
              (unsigned long long)((tb - fb) >> 20),
              (unsigned long long)(tb >> 20));

  double sum_prefill = 0, sum_toks = 0;
  int ok = 0;
  for (const auto& pr : results) {
    sum_prefill += pr.prefill_ms;
    sum_toks += pr.tok_s;
    if (pr.miss == 0 && pr.h2d_calls == 0) ++ok;
  }
  std::printf("avg prefill (TTFT):  %.1f ms\n", sum_prefill / n_repeats);
  std::printf("avg decode tok/s:    %.2f\n", sum_toks / n_repeats);
  for (int i = 0; i < n_repeats; ++i)
    std::printf("  pass %d:  %.2f tok/s  prefill %.1f ms  hit %.1f%%\n", i + 1,
                results[i].tok_s, results[i].prefill_ms,
                (results[i].hits + results[i].miss)
                    ? 100.0 * results[i].hits /
                          (results[i].hits + results[i].miss)
                    : 0.0);

  long sel_all = total_hits + total_miss;
  std::printf("overall hit rate:    %ld / %ld (%.2f%%)\n", total_hits, sel_all,
              sel_all ? 100.0 * total_hits / sel_all : 0.0);
  std::printf("H2D during repeats:  %d calls (want 0)\n", total_h2d);
  std::printf("pure-VRAM passes:    %d / %d\n", ok, n_repeats);
  std::printf("\nVERDICT: %s\n",
              (ok == n_repeats)
                  ? "YES — experts pinned first; all 3 \"hi\" runs pure VRAM "
                    "(3B-style resident path)"
                  : "PARTIAL — some misses or H2D on re-runs (route expanded "
                    "beyond pin set)");
  // Separate synchronized diagnostic run: excluded from headline tok/s.
  std::printf("\n=== DETAILED PROFILE (separate pinned run; not headline tok/s) ===\n");
  {
    Forward fprof(dm, c, max_ctx, man);
    StepProf prof;
    fprof.prof = &prof;
    fprof.gpu_dispatch = use_gpu_dispatch;
    fprof.use_cuda_graph = false;  // graphs off while profiling
    long phits = 0, pmiss = 0;
    fprof.ensure_experts = make_probe(&phits, &pmiss);
    int hcalls_before = h2d_calls;
    uint64_t hbytes_before = h2d_bytes;
    run_hi(fprof, c, detok, prompt, n_gen, &h2d_calls, &h2d_bytes);
    const int ntok = (int)prof.tok_ms.size();
    double avg_wall = 0.0;
    for (double x : prof.tok_ms) avg_wall += x;
    if (ntok) avg_wall /= ntok;
    auto per_tok = [ntok](double x) { return ntok ? x / ntok : 0.0; };
    std::printf("profiled tokens: %d  wall mean %.3f ms  p50 %.3f  p95 %.3f  p99 %.3f\n",
                ntok, avg_wall, pct(prof.tok_ms, 50), pct(prof.tok_ms, 95),
                pct(prof.tok_ms, 99));
    std::printf("phase ms/token: embed %.3f | attention %.3f | router %.3f | experts %.3f | final %.3f\n",
                per_tok(prof.embed_ms), per_tok(prof.attn_ms),
                per_tok(prof.router_ms), per_tok(prof.experts_ms),
                per_tok(prof.final_ms));
    std::printf("\n--- DEEP SUB-PHASE ms/token (sync-inflated; relative shares) ---\n");
    std::printf("ATTENTION breakdown:\n");
    std::printf("  attn_norm          %.3f\n", per_tok(prof.attn_norm_ms));
    std::printf("  quantize_attn      %.3f\n", per_tok(prof.quantize_attn_ms));
    std::printf("  q_proj             %.3f\n", per_tok(prof.q_proj_ms));
    std::printf("  k_proj             %.3f\n", per_tok(prof.k_proj_ms));
    std::printf("  v_proj             %.3f\n", per_tok(prof.v_proj_ms));
    std::printf("  rope+q/k_norm      %.3f\n", per_tok(prof.rope_ms));
    std::printf("  kv_store           %.3f\n", per_tok(prof.kv_store_ms));
    std::printf("  gqa_attn           %.3f\n", per_tok(prof.gqa_ms));
    std::printf("  attn_out+residual  %.3f\n", per_tok(prof.attn_out_ms));
    std::printf("ROUTER breakdown:\n");
    std::printf("  ffn_norm (pre)     %.3f  (counted in experts coarse)\n",
                per_tok(prof.ffn_norm_ms));
    std::printf("  router_gemv        %.3f\n", per_tok(prof.router_gemv_ms));
    std::printf("  router_topk        %.3f\n", per_tok(prof.router_topk_ms));
    std::printf("EXPERTS breakdown:\n");
    std::printf("  quantize_model     %.3f\n", per_tok(prof.quantize_model_ms));
    std::printf("  gate_up_silu       %.3f\n", per_tok(prof.gate_up_ms));
    std::printf("  quantize_ff        %.3f\n", per_tok(prof.quantize_ff_ms));
    std::printf("  down               %.3f\n", per_tok(prof.down_ms));
    std::printf("  residual           %.3f\n", per_tok(prof.residual_ms));
    std::printf("FINAL breakdown:\n");
    std::printf("  final_norm         %.3f\n", per_tok(prof.final_norm_ms));
    std::printf("  logits_gemv        %.3f\n", per_tok(prof.logits_ms));
    std::printf("  sample+D2H         %.3f\n", per_tok(prof.sample_ms));
    const double deep_sum =
        per_tok(prof.embed_ms) + per_tok(prof.attn_norm_ms) +
        per_tok(prof.quantize_attn_ms) + per_tok(prof.q_proj_ms) +
        per_tok(prof.k_proj_ms) + per_tok(prof.v_proj_ms) +
        per_tok(prof.rope_ms) + per_tok(prof.kv_store_ms) +
        per_tok(prof.gqa_ms) + per_tok(prof.attn_out_ms) +
        per_tok(prof.ffn_norm_ms) + per_tok(prof.router_gemv_ms) +
        per_tok(prof.router_topk_ms) + per_tok(prof.quantize_model_ms) +
        per_tok(prof.gate_up_ms) + per_tok(prof.quantize_ff_ms) +
        per_tok(prof.down_ms) + per_tok(prof.residual_ms) +
        per_tok(prof.final_norm_ms) + per_tok(prof.logits_ms) +
        per_tok(prof.sample_ms);
    std::printf("deep_sum ms/token:   %.3f  (vs wall mean %.3f; gap=sync/overhead)\n",
                deep_sum, avg_wall);
    std::printf("host sync ms/token: router %.3f | logits/sample %.3f\n",
                per_tok(prof.router_sync_ms), per_tok(prof.logits_sync_ms));
    FILE* deep = std::fopen("build/profile_deep.csv", "wb");
    if (deep) {
      std::fprintf(deep,
                   "bucket,ms_per_token,pct_of_deep_sum\n");
      auto row = [&](const char* name, double ms) {
        std::fprintf(deep, "%s,%.6f,%.2f\n", name, ms,
                     deep_sum > 0 ? 100.0 * ms / deep_sum : 0.0);
      };
      row("embed", per_tok(prof.embed_ms));
      row("attn_norm", per_tok(prof.attn_norm_ms));
      row("quantize_attn", per_tok(prof.quantize_attn_ms));
      row("q_proj", per_tok(prof.q_proj_ms));
      row("k_proj", per_tok(prof.k_proj_ms));
      row("v_proj", per_tok(prof.v_proj_ms));
      row("rope", per_tok(prof.rope_ms));
      row("kv_store", per_tok(prof.kv_store_ms));
      row("gqa_attn", per_tok(prof.gqa_ms));
      row("attn_out", per_tok(prof.attn_out_ms));
      row("ffn_norm", per_tok(prof.ffn_norm_ms));
      row("router_gemv", per_tok(prof.router_gemv_ms));
      row("router_topk", per_tok(prof.router_topk_ms));
      row("quantize_model", per_tok(prof.quantize_model_ms));
      row("gate_up", per_tok(prof.gate_up_ms));
      row("quantize_ff", per_tok(prof.quantize_ff_ms));
      row("down", per_tok(prof.down_ms));
      row("residual", per_tok(prof.residual_ms));
      row("final_norm", per_tok(prof.final_norm_ms));
      row("logits", per_tok(prof.logits_ms));
      row("sample", per_tok(prof.sample_ms));
      std::fclose(deep);
      std::printf("deep sub-phase CSV: build/profile_deep.csv\n");
    }
    std::printf("transfers: H2D %llu B/%d calls | D2H %llu B/%d calls | D2D %llu B/%d calls\n",
                (unsigned long long)(h2d_bytes - hbytes_before),
                h2d_calls - hcalls_before, (unsigned long long)prof.d2h_bytes,
                prof.d2h_calls, (unsigned long long)prof.d2d_bytes, prof.d2d_calls);
    std::printf("quant weight reads: %.3f GiB total (%.3f GiB/token)\n",
                prof.dequant_bytes_in / 1073741824.0,
                ntok ? prof.dequant_bytes_in / 1073741824.0 / ntok : 0.0);
    std::printf("experts: hits %ld misses %ld evictions %ld | peak VRAM %.1f MiB\n",
                prof.expert_hits, prof.expert_miss, prof.evictions,
                prof.vram_peak_used / 1048576.0);
    std::printf("CUDA launches: %ld total (%.1f/token)\n", prof.launches.total,
                ntok ? (double)prof.launches.total / ntok : 0.0);
    for (const auto& kv : prof.launches.calls)
      std::printf("  %-28s %ld (%.1f/token)\n", kv.first.c_str(), kv.second,
                  ntok ? (double)kv.second / ntok : 0.0);
    FILE* trace = std::fopen("build/profile_trace.csv", "wb");
    if (trace) {
      std::fprintf(trace, "pos,input_token,wall_ms,embed_gpu_ms,attention_gpu_ms,router_gpu_ms,experts_gpu_ms,final_gpu_ms,phase_total_ms,unattributed_gap_ms,h2d_bytes,d2h_bytes,d2d_bytes,vram_used_bytes\n");
      for (const auto& t : prof.trace) {
        const double gpu = t.embed_gpu_ms + t.attention_gpu_ms + t.router_gpu_ms + t.experts_gpu_ms + t.final_gpu_ms;
        std::fprintf(trace, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%llu,%llu,%llu,%llu\n",
                     t.pos, t.input_token, t.wall_ms, t.embed_gpu_ms,
                     t.attention_gpu_ms, t.router_gpu_ms, t.experts_gpu_ms,
                     t.final_gpu_ms, gpu, t.wall_ms - gpu,
                     (unsigned long long)t.h2d_bytes, (unsigned long long)t.d2h_bytes,
                     (unsigned long long)t.d2d_bytes, (unsigned long long)t.vram_used);
      }
      std::fclose(trace);
      std::printf("per-token real trace: build/profile_trace.csv\n");
    }
  }
  return 0;
}
