// MoEx clean-room. Device residency implementation.
#include "cuda/device_model.h"

#include "cuda/cuda_common.h"

namespace moex {

DeviceModel::~DeviceModel() {
  for (void* p : allocs_) {
    if (p) cudaFree(p);
  }
}

DeviceTensor DeviceModel::upload_tensor_(const gguf::TensorInfo* t,
                                         const uint8_t* host_base, uint64_t rows,
                                         uint64_t cols) {
  DeviceTensor dt;
  dt.nbytes = t->nbytes;
  dt.type = t->type;
  dt.rows = rows;
  dt.cols = cols;
  void* d = nullptr;
  MOEX_CUDA(cudaMalloc(&d, t->nbytes));
  MOEX_CUDA(cudaMemcpy(d, host_base + t->file_offset, t->nbytes,
                             cudaMemcpyHostToDevice));
  dt.dptr = d;
  allocs_.push_back(d);
  bytes_resident_ += t->nbytes;
  return dt;
}

DeviceTensor DeviceModel::upload_slice_(const ByteSlice& s,
                                        const uint8_t* host_base) {
  DeviceTensor dt;
  dt.nbytes = s.nbytes;
  dt.type = s.type;
  dt.rows = s.rows;
  dt.cols = s.cols;
  void* d = nullptr;
  MOEX_CUDA(cudaMalloc(&d, s.nbytes));
  MOEX_CUDA(cudaMemcpy(d, host_base + s.file_offset, s.nbytes,
                             cudaMemcpyHostToDevice));
  dt.dptr = d;
  allocs_.push_back(d);
  bytes_resident_ += s.nbytes;
  return dt;
}

uint64_t DeviceModel::upload_static(const gguf::Model& gguf, const Manifest& man,
                                    const uint8_t* host_base, std::string* err) {
  (void)gguf;
  const ModelConfig& c = man.config();
  n_experts_ = c.n_experts;
  const uint64_t start = bytes_resident_;

  try {
    // Global tensors.
    token_embd_ = upload_tensor_(man.token_embd(), host_base, c.vocab_size,
                                 c.d_model);
    output_norm_ = upload_tensor_(man.output_norm(), host_base, 1, c.d_model);
    output_ = upload_tensor_(man.output(), host_base, c.vocab_size, c.d_model);

    layers_.resize(c.n_layers);
    for (uint32_t l = 0; l < c.n_layers; ++l) {
      const LayerTensors& lt = man.layers()[l];
      DeviceLayer& dl = layers_[l];
      dl.attn_norm = upload_tensor_(lt.attn_norm, host_base, 1, c.d_model);
      dl.attn_q = upload_tensor_(lt.attn_q, host_base, c.q_dim(), c.d_model);
      dl.attn_k = upload_tensor_(lt.attn_k, host_base, c.kv_dim(), c.d_model);
      dl.attn_v = upload_tensor_(lt.attn_v, host_base, c.kv_dim(), c.d_model);
      dl.attn_output =
          upload_tensor_(lt.attn_output, host_base, c.d_model, c.q_dim());
      dl.attn_q_norm = upload_tensor_(lt.attn_q_norm, host_base, 1, c.head_dim);
      dl.attn_k_norm = upload_tensor_(lt.attn_k_norm, host_base, 1, c.head_dim);
      dl.ffn_norm = upload_tensor_(lt.ffn_norm, host_base, 1, c.d_model);
      dl.router =
          upload_tensor_(lt.ffn_gate_inp, host_base, c.n_experts, c.d_model);
    }
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return 0;
  }
  return bytes_resident_ - start;
}

uint64_t DeviceModel::upload_experts(
    const gguf::Model& gguf, const Manifest& man, const uint8_t* host_base,
    const std::vector<std::vector<uint32_t>>& which, std::string* err) {
  (void)gguf;
  const ModelConfig& c = man.config();
  const uint64_t start = bytes_resident_;
  if (experts_.empty()) {
    experts_.resize(static_cast<size_t>(c.n_layers) * c.n_experts);
  }
  try {
    for (uint32_t l = 0; l < c.n_layers && l < which.size(); ++l) {
      for (uint32_t e : which[l]) {
        const ExpertBundle& b = man.expert(l, e);
        DeviceExpert& de = experts_[static_cast<size_t>(l) * c.n_experts + e];
        if (de.resident) continue;
        de.gate = upload_slice_(b.gate, host_base);
        de.up = upload_slice_(b.up, host_base);
        de.down = upload_slice_(b.down, host_base);
        de.resident = true;
      }
    }
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return 0;
  }
  return bytes_resident_ - start;
}

}  // namespace moex
