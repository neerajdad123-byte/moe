// MoEx clean-room. Manifest: canonicalize a parsed GGUF into the executable
// model description MoEx runs against — config, the static-resident tensor
// set, the ExpertKey -> stacked-slice map, and a byte ledger / feasibility
// gate (design doc §2, G0). Nothing here reads tensor payloads; it computes
// byte ranges and plans memory.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf/gguf.h"

namespace moex {

// Model hyperparameters, read from qwen3moe.* metadata. All required; a
// missing key fails the build (we do not guess architecture facts).
struct ModelConfig {
  uint32_t n_layers = 0;         // block_count
  uint32_t n_experts = 0;        // expert_count
  uint32_t n_experts_used = 0;   // expert_used_count (top-k)
  uint32_t d_model = 0;          // embedding_length
  uint32_t n_head = 0;           // attention.head_count
  uint32_t n_head_kv = 0;        // attention.head_count_kv
  uint32_t head_dim = 0;         // attention.key_length
  uint32_t d_ff_expert = 0;      // expert_feed_forward_length
  uint32_t d_ff_dense = 0;       // feed_forward_length (dense, unused by MoE blocks)
  uint32_t n_ctx_train = 0;      // context_length
  uint32_t vocab_size = 0;       // from token_embd dim / tokenizer
  float rms_eps = 1e-6f;         // attention.layer_norm_rms_epsilon
  float rope_base = 1e6f;        // rope.freq_base

  uint32_t kv_dim() const { return n_head_kv * head_dim; }
  uint32_t q_dim() const { return n_head * head_dim; }
};

// A contiguous byte range within one GGUF tensor (used for expert slices).
struct ByteSlice {
  const gguf::TensorInfo* tensor = nullptr;  // parent stacked tensor
  uint64_t file_offset = 0;                  // absolute offset of this slice
  uint64_t nbytes = 0;
  gguf::GgmlType type = gguf::GgmlType::UNKNOWN;
  uint64_t rows = 0;   // logical [rows x cols] of the 2D expert matrix
  uint64_t cols = 0;
};

// One expert's three matrices, sliced out of the per-layer stacked tensors.
// (gate/up are Q4_K, down is Q6_K for this blob — see moex-blob-structure.)
struct ExpertBundle {
  uint32_t layer = 0;
  uint32_t expert = 0;
  ByteSlice gate;  // [d_ff_expert x d_model]
  ByteSlice up;    // [d_ff_expert x d_model]
  ByteSlice down;  // [d_model x d_ff_expert]
  uint64_t total_bytes() const { return gate.nbytes + up.nbytes + down.nbytes; }
};

// Per-layer non-expert (attention + norms + router) tensors, all static-resident.
struct LayerTensors {
  const gguf::TensorInfo* attn_norm = nullptr;
  const gguf::TensorInfo* attn_q = nullptr;
  const gguf::TensorInfo* attn_k = nullptr;
  const gguf::TensorInfo* attn_v = nullptr;
  const gguf::TensorInfo* attn_output = nullptr;
  const gguf::TensorInfo* attn_q_norm = nullptr;
  const gguf::TensorInfo* attn_k_norm = nullptr;
  const gguf::TensorInfo* ffn_norm = nullptr;
  const gguf::TensorInfo* ffn_gate_inp = nullptr;  // router
  const gguf::TensorInfo* ffn_gate_exps = nullptr;
  const gguf::TensorInfo* ffn_up_exps = nullptr;
  const gguf::TensorInfo* ffn_down_exps = nullptr;
};

// The VRAM byte ledger and feasibility verdict (design doc §2).
struct MemoryLedger {
  uint64_t vram_total = 0;
  uint64_t vram_usable = 0;      // after CUDA context + guard band
  uint64_t static_bytes = 0;     // non-expert resident weights
  uint64_t kv_bytes = 0;         // for chosen context bucket
  uint64_t activation_bytes = 0; // workspaces
  uint64_t overhead_bytes = 0;   // graph/event/pointer tables + fragmentation
  uint64_t expert_arena = 0;     // what remains for expert cache
  uint64_t bundle_min = 0, bundle_med = 0, bundle_max = 0;
  uint64_t feasibility_floor = 0;  // 8 live + 1 reactive + fragmentation
  uint32_t arena_capacity_experts = 0;
  bool feasible = false;
  uint32_t ctx_bucket = 0;
};

class Manifest {
 public:
  // Build from an already-open GGUF model. Fills `err` and returns false on any
  // missing required tensor/metadata. Does not read tensor payloads.
  bool build(const gguf::Model& m, std::string* err);

  const ModelConfig& config() const { return cfg_; }
  const std::vector<LayerTensors>& layers() const { return layers_; }
  const ExpertBundle& expert(uint32_t layer, uint32_t e) const {
    return experts_[static_cast<size_t>(layer) * cfg_.n_experts + e];
  }
  const std::vector<const gguf::TensorInfo*>& static_tensors() const {
    return static_tensors_;
  }
  uint64_t static_bytes() const { return static_bytes_; }

  // Global tensors.
  const gguf::TensorInfo* token_embd() const { return token_embd_; }
  const gguf::TensorInfo* output_norm() const { return output_norm_; }
  const gguf::TensorInfo* output() const { return output_; }

  // Compute the byte ledger + feasibility for a context bucket and VRAM budget.
  MemoryLedger plan(uint64_t vram_total, uint32_t ctx_bucket) const;

 private:
  ModelConfig cfg_;
  std::vector<LayerTensors> layers_;
  std::vector<ExpertBundle> experts_;  // layer-major, size n_layers*n_experts
  std::vector<const gguf::TensorInfo*> static_tensors_;
  uint64_t static_bytes_ = 0;
  const gguf::TensorInfo* token_embd_ = nullptr;
  const gguf::TensorInfo* output_norm_ = nullptr;
  const gguf::TensorInfo* output_ = nullptr;
};

}  // namespace moex
