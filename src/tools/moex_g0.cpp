// MoEx G0 tool: parse the blob, build the manifest, print config + expert
// slice sanity + the VRAM byte ledger / feasibility verdict.
#include <cstdio>

#include "gguf/gguf.h"
#include "model/manifest.h"

using namespace moex;

static double GiB(uint64_t b) { return b / (1024.0 * 1024 * 1024); }
static double MiB(uint64_t b) { return b / (1024.0 * 1024); }

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: moex_g0 <file.gguf> [vram_MiB] [ctx]\n"); return 2; }
  uint64_t vram = (argc > 2 ? (uint64_t)atoll(argv[2]) : 6141) * 1024ull * 1024;
  uint32_t ctx = argc > 3 ? (uint32_t)atoi(argv[3]) : 4096;

  gguf::Model m;
  std::string err;
  if (!m.open(argv[1], &err)) { std::fprintf(stderr, "parse: %s\n", err.c_str()); return 1; }

  Manifest man;
  if (!man.build(m, &err)) { std::fprintf(stderr, "manifest: %s\n", err.c_str()); return 1; }

  const ModelConfig& c = man.config();
  std::printf("=== config ===\n");
  std::printf("layers=%u experts=%u top_k=%u d_model=%u\n", c.n_layers, c.n_experts,
              c.n_experts_used, c.d_model);
  std::printf("n_head=%u n_head_kv=%u head_dim=%u q_dim=%u kv_dim=%u\n", c.n_head,
              c.n_head_kv, c.head_dim, c.q_dim(), c.kv_dim());
  std::printf("d_ff_expert=%u vocab=%u ctx_train=%u rms_eps=%g rope_base=%g\n",
              c.d_ff_expert, c.vocab_size, c.n_ctx_train, c.rms_eps, c.rope_base);

  // Expert slice sanity: layer 0 expert 0 and expert 127; check contiguity and
  // that expert e offset == base + e*stride, and last slice stays in-tensor.
  std::printf("\n=== expert slice check (layer 0) ===\n");
  const ExpertBundle& e0 = man.expert(0, 0);
  const ExpertBundle& eL = man.expert(0, c.n_experts - 1);
  std::printf("e0   gate off=%llu bytes=%llu | up off=%llu | down off=%llu bytes=%llu\n",
              (unsigned long long)e0.gate.file_offset, (unsigned long long)e0.gate.nbytes,
              (unsigned long long)e0.up.file_offset,
              (unsigned long long)e0.down.file_offset, (unsigned long long)e0.down.nbytes);
  std::printf("e127 gate off=%llu | bundle=%.3f MiB\n",
              (unsigned long long)eL.gate.file_offset, MiB(e0.total_bytes()));
  // Verify last down slice end <= tensor end.
  const gguf::TensorInfo* dt = man.layers()[0].ffn_down_exps;
  uint64_t last_end = eL.down.file_offset + eL.down.nbytes;
  uint64_t tensor_end = dt->file_offset + dt->nbytes;
  std::printf("down last_slice_end=%llu tensor_end=%llu %s\n",
              (unsigned long long)last_end, (unsigned long long)tensor_end,
              last_end == tensor_end ? "OK (exact)" : "MISMATCH");

  std::printf("\n=== byte ledger (vram=%.2f GiB, ctx=%u) ===\n", GiB(vram), ctx);
  MemoryLedger L = man.plan(vram, ctx);
  std::printf("vram_usable      %.3f GiB\n", GiB(L.vram_usable));
  std::printf("static weights   %.3f GiB\n", GiB(L.static_bytes));
  std::printf("kv cache         %.3f GiB\n", GiB(L.kv_bytes));
  std::printf("activations      %.1f MiB\n", MiB(L.activation_bytes));
  std::printf("overhead/frag    %.1f MiB\n", MiB(L.overhead_bytes));
  std::printf("-> expert arena  %.3f GiB\n", GiB(L.expert_arena));
  std::printf("bundle min/med/max = %.2f / %.2f / %.2f MiB\n", MiB(L.bundle_min),
              MiB(L.bundle_med), MiB(L.bundle_max));
  std::printf("arena holds ~%u experts\n", L.arena_capacity_experts);
  std::printf("feasibility floor %.3f GiB (8 live + reactive + frag)\n",
              GiB(L.feasibility_floor));
  std::printf("VERDICT: %s\n", L.feasible ? "FEASIBLE" : "target configuration infeasible");
  return L.feasible ? 0 : 3;
}
