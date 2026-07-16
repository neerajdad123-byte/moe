// MoEx clean-room. Manifest builder + byte ledger (G0).
#include "model/manifest.h"

#include <algorithm>
#include <cstdio>

namespace moex {

namespace {

// Bytes for `elems` elements of a quantized/float type, using the type's block.
bool elems_to_bytes(gguf::GgmlType t, uint64_t elems, uint64_t* out) {
  uint64_t bb, be;
  if (!gguf::ggml_type_block(t, &bb, &be)) return false;
  if (elems % be != 0) return false;  // must be block-aligned
  *out = (elems / be) * bb;
  return true;
}

const gguf::TensorInfo* req(const gguf::Model& m, const std::string& name,
                            std::string* err) {
  const gguf::TensorInfo* t = m.tensor(name);
  if (!t && err) *err = "missing required tensor: " + name;
  return t;
}

}  // namespace

bool Manifest::build(const gguf::Model& m, std::string* err) {
  const std::string arch = m.get_str("general.architecture", "");
  if (arch != "qwen3moe") {
    if (err) *err = "unsupported architecture '" + arch + "' (expected qwen3moe)";
    return false;
  }

  // --- config from metadata (all required) ---------------------------------
  auto getu = [&](const char* k, uint32_t* dst) -> bool {
    const gguf::Value* v = m.find(k);
    if (!v || !v->is_int()) { if (err) *err = std::string("missing kv: ") + k; return false; }
    *dst = static_cast<uint32_t>(v->as_u64());
    return true;
  };
  if (!getu("qwen3moe.block_count", &cfg_.n_layers)) return false;
  if (!getu("qwen3moe.expert_count", &cfg_.n_experts)) return false;
  if (!getu("qwen3moe.expert_used_count", &cfg_.n_experts_used)) return false;
  if (!getu("qwen3moe.embedding_length", &cfg_.d_model)) return false;
  if (!getu("qwen3moe.attention.head_count", &cfg_.n_head)) return false;
  if (!getu("qwen3moe.attention.head_count_kv", &cfg_.n_head_kv)) return false;
  if (!getu("qwen3moe.attention.key_length", &cfg_.head_dim)) return false;
  if (!getu("qwen3moe.expert_feed_forward_length", &cfg_.d_ff_expert)) return false;
  getu("qwen3moe.feed_forward_length", &cfg_.d_ff_dense);  // optional-ish
  if (!getu("qwen3moe.context_length", &cfg_.n_ctx_train)) return false;
  cfg_.rms_eps = static_cast<float>(
      m.get_f64("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6));
  cfg_.rope_base = static_cast<float>(m.get_f64("qwen3moe.rope.freq_base", 1e6));

  // --- global tensors ------------------------------------------------------
  token_embd_ = req(m, "token_embd.weight", err);
  output_norm_ = req(m, "output_norm.weight", err);
  if (!token_embd_ || !output_norm_) return false;
  // Some GGUFs tie lm_head to token_embd; this blob has a separate output.weight.
  output_ = m.tensor("output.weight");
  if (!output_) output_ = token_embd_;  // tied fallback

  // vocab = slow dim of token_embd [d_model, vocab]
  if (token_embd_->dims.size() != 2) {
    if (err) *err = "token_embd has unexpected rank";
    return false;
  }
  cfg_.vocab_size = static_cast<uint32_t>(token_embd_->dims[1]);

  // --- per-layer tensors + expert slices -----------------------------------
  layers_.resize(cfg_.n_layers);
  experts_.resize(static_cast<size_t>(cfg_.n_layers) * cfg_.n_experts);

  for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
    const std::string p = "blk." + std::to_string(L) + ".";
    LayerTensors& lt = layers_[L];
    lt.attn_norm = req(m, p + "attn_norm.weight", err);
    lt.attn_q = req(m, p + "attn_q.weight", err);
    lt.attn_k = req(m, p + "attn_k.weight", err);
    lt.attn_v = req(m, p + "attn_v.weight", err);
    lt.attn_output = req(m, p + "attn_output.weight", err);
    lt.attn_q_norm = req(m, p + "attn_q_norm.weight", err);
    lt.attn_k_norm = req(m, p + "attn_k_norm.weight", err);
    lt.ffn_norm = req(m, p + "ffn_norm.weight", err);
    lt.ffn_gate_inp = req(m, p + "ffn_gate_inp.weight", err);
    lt.ffn_gate_exps = req(m, p + "ffn_gate_exps.weight", err);
    lt.ffn_up_exps = req(m, p + "ffn_up_exps.weight", err);
    lt.ffn_down_exps = req(m, p + "ffn_down_exps.weight", err);
    if (!lt.attn_norm || !lt.attn_q || !lt.attn_k || !lt.attn_v ||
        !lt.attn_output || !lt.attn_q_norm || !lt.attn_k_norm || !lt.ffn_norm ||
        !lt.ffn_gate_inp || !lt.ffn_gate_exps || !lt.ffn_up_exps ||
        !lt.ffn_down_exps)
      return false;

    // Stacked expert tensors. GGUF dims are fast-first:
    //   gate/up : [d_model, d_ff_expert, n_experts]
    //   down    : [d_ff_expert, d_model, n_experts]
    // Expert e is a contiguous slab of (dim0*dim1) elements at index e.
    auto slice = [&](const gguf::TensorInfo* t, uint32_t e, uint64_t rows,
                     uint64_t cols, ByteSlice* bs) -> bool {
      if (t->dims.size() != 3 || t->dims[2] != cfg_.n_experts) {
        if (err) *err = t->name + ": unexpected stacked expert shape";
        return false;
      }
      uint64_t per_expert_elems = t->dims[0] * t->dims[1];
      uint64_t nb;
      if (!elems_to_bytes(t->type, per_expert_elems, &nb)) {
        if (err) *err = t->name + ": non-block-aligned expert slice";
        return false;
      }
      bs->tensor = t;
      bs->type = t->type;
      bs->nbytes = nb;
      bs->file_offset = t->file_offset + static_cast<uint64_t>(e) * nb;
      bs->rows = rows;
      bs->cols = cols;
      return true;
    };

    for (uint32_t e = 0; e < cfg_.n_experts; ++e) {
      ExpertBundle& b = experts_[static_cast<size_t>(L) * cfg_.n_experts + e];
      b.layer = L;
      b.expert = e;
      if (!slice(lt.ffn_gate_exps, e, cfg_.d_ff_expert, cfg_.d_model, &b.gate)) return false;
      if (!slice(lt.ffn_up_exps, e, cfg_.d_ff_expert, cfg_.d_model, &b.up)) return false;
      if (!slice(lt.ffn_down_exps, e, cfg_.d_model, cfg_.d_ff_expert, &b.down)) return false;
    }
  }

  // --- static-resident tensor set (everything not an expert stack) ---------
  static_tensors_.push_back(token_embd_);
  static_tensors_.push_back(output_norm_);
  if (output_ != token_embd_) static_tensors_.push_back(output_);
  for (const LayerTensors& lt : layers_) {
    const gguf::TensorInfo* arr[] = {lt.attn_norm, lt.attn_q, lt.attn_k,
                                     lt.attn_v, lt.attn_output, lt.attn_q_norm,
                                     lt.attn_k_norm, lt.ffn_norm, lt.ffn_gate_inp};
    for (auto* t : arr) static_tensors_.push_back(t);
  }
  static_bytes_ = 0;
  for (auto* t : static_tensors_) static_bytes_ += t->nbytes;

  return true;
}

MemoryLedger Manifest::plan(uint64_t vram_total, uint32_t ctx_bucket) const {
  MemoryLedger L{};
  L.vram_total = vram_total;
  L.ctx_bucket = ctx_bucket;

  // CUDA context + guard band: reserve ~600 MiB (measured refinement later).
  const uint64_t guard = 600ull * 1024 * 1024;
  L.vram_usable = vram_total > guard ? vram_total - guard : 0;

  L.static_bytes = static_bytes_;

  // KV cache, FP16, both K and V: n_layers * 2 * n_head_kv * head_dim * 2 bytes
  // per token, times context bucket.
  uint64_t kv_per_tok =
      static_cast<uint64_t>(cfg_.n_layers) * 2 * cfg_.kv_dim() * 2;
  L.kv_bytes = kv_per_tok * ctx_bucket;

  // Activation/workspace: a few MB of f32 scratch (hidden, ff, logits buffers).
  L.activation_bytes = 64ull * 1024 * 1024;
  // Graph/event/pointer tables + allocator fragmentation reserve.
  L.overhead_bytes = 128ull * 1024 * 1024;

  uint64_t reserved =
      L.static_bytes + L.kv_bytes + L.activation_bytes + L.overhead_bytes;
  L.expert_arena = L.vram_usable > reserved ? L.vram_usable - reserved : 0;

  // Expert bundle size distribution (all layers/experts).
  std::vector<uint64_t> sizes;
  sizes.reserve(experts_.size());
  for (const auto& b : experts_) sizes.push_back(b.total_bytes());
  if (!sizes.empty()) {
    std::sort(sizes.begin(), sizes.end());
    L.bundle_min = sizes.front();
    L.bundle_max = sizes.back();
    L.bundle_med = sizes[sizes.size() / 2];
  }

  // Feasibility floor: top-k live + one reactive bundle + fragmentation reserve.
  uint64_t frag = L.bundle_max;  // one-bundle fragmentation reserve
  L.feasibility_floor =
      static_cast<uint64_t>(cfg_.n_experts_used) * L.bundle_max + L.bundle_max + frag;
  L.arena_capacity_experts =
      L.bundle_max ? static_cast<uint32_t>(L.expert_arena / L.bundle_max) : 0;
  L.feasible = L.expert_arena >= L.feasibility_floor;
  return L;
}

}  // namespace moex
