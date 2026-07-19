// MoEx clean-room. Device residency implementation.
#include "cuda/device_model.h"

#include <algorithm>
#include <cstring>

#include "cuda/cuda_common.h"  // GpuExpertPtr is declared in device_model.h

namespace moex {

DeviceModel::~DeviceModel() {
  for (void* p : allocs_) {
    if (p) cudaFree(p);
  }
  if (d_ptrs_) cudaFree(d_ptrs_);
  if (arena_mode_) {
    // All expert dptrs point inside the single arena allocation.
    cudaDeviceSynchronize();  // let in-flight staging copies drain first
    for (auto& e : seg_ev_) cudaEventDestroy(e);
    for (auto& e : batch_ev_) cudaEventDestroy(e);
    for (auto& e : fence_ev_) cudaEventDestroy(e);
    if (copy_stream_) cudaStreamDestroy(copy_stream_);
    if (pin_buf_) cudaFreeHost(pin_buf_);
    if (arena_) cudaFree(arena_);
  } else {
    // Expert bundles uploaded via ensure_bounded() aren't in allocs_ (their
    // dptrs are freed directly on eviction); free whatever's still resident.
    for (auto& de : experts_) {
      if (!de.resident) continue;
      if (de.gate.dptr) cudaFree(de.gate.dptr);
      if (de.up.dptr) cudaFree(de.up.dptr);
      if (de.down.dptr) cudaFree(de.down.dptr);
    }
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

DeviceTensor DeviceModel::upload_slice_untracked_(const ByteSlice& s,
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
  bytes_resident_ += s.nbytes;
  return dt;
}

uint64_t DeviceModel::upload_static(const gguf::Model& gguf, const Manifest& man,
                                    const uint8_t* host_base, std::string* err) {
  (void)gguf;
  const ModelConfig& c = man.config();
  n_experts_ = c.n_experts;
  // Pre-size expert bookkeeping (all non-resident) so `expert(l,e)` is a
  // safe query — including from a profiler attached before the *first*
  // ensure_bounded()/upload_experts() call — instead of lazily resizing on
  // first upload, which left a window where the vector was still empty.
  experts_.assign((size_t)c.n_layers * c.n_experts, DeviceExpert{});
  // GPU mirror of the pointer table (all-zero = not resident, null ptrs),
  // for dispatch_gather_kernel's zero-host-round-trip fast path.
  const size_t n_flat = experts_.size();
  if (cudaMalloc(&d_ptrs_, n_flat * sizeof(GpuExpertPtr)) == cudaSuccess) {
    cudaMemset(d_ptrs_, 0, n_flat * sizeof(GpuExpertPtr));
  } else {
    d_ptrs_ = nullptr;
  }
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
        sync_gpu_ptr_(static_cast<size_t>(l) * c.n_experts + e, de);
      }
    }
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
    return 0;
  }
  return bytes_resident_ - start;
}

uint32_t DeviceModel::init_arena(const Manifest& man, uint32_t capacity,
                                 uint64_t vram_reserve_bytes, std::string* err) {
  const ModelConfig& c = man.config();
  if (experts_.empty()) experts_.assign((size_t)c.n_layers * c.n_experts, DeviceExpert{});

  // Fixed intra-slot layout from the largest gate/up/down slice in the model
  // (bundle sizes are uniform for this blob except a handful of layers with a
  // smaller down type; slots are sized for the max so any expert fits any slot).
  uint64_t gmax = 0, umax = 0, dmax = 0;
  for (uint32_t l = 0; l < c.n_layers; ++l)
    for (uint32_t e = 0; e < c.n_experts; ++e) {
      const ExpertBundle& b = man.expert(l, e);
      gmax = std::max(gmax, b.gate.nbytes);
      umax = std::max(umax, b.up.nbytes);
      dmax = std::max(dmax, b.down.nbytes);
    }
  auto align = [](uint64_t x) { return (x + 255) & ~255ull; };
  gate_off_ = 0;
  up_off_ = align(gmax);
  down_off_ = up_off_ + align(umax);
  slot_stride_ = down_off_ + align(dmax);

  if (capacity == 0) {  // auto-size from live free VRAM
    size_t freeB = 0, totB = 0;
    if (cudaMemGetInfo(&freeB, &totB) != cudaSuccess) {
      if (err) *err = "cudaMemGetInfo failed";
      return 0;
    }
    if (freeB <= vram_reserve_bytes) {
      if (err) *err = "free VRAM below reserve; nothing left for expert arena";
      return 0;
    }
    capacity = (uint32_t)((freeB - vram_reserve_bytes) / slot_stride_);
  }
  if (capacity == 0) {
    if (err) *err = "arena capacity computed as 0";
    return 0;
  }

  if (cudaMalloc(&arena_, (uint64_t)capacity * slot_stride_) != cudaSuccess) {
    if (err) *err = "cudaMalloc of expert arena failed";
    return 0;
  }
  n_segs_ = 32;
  if (cudaHostAlloc(&pin_buf_, (uint64_t)n_segs_ * slot_stride_,
                    cudaHostAllocDefault) != cudaSuccess) {
    cudaFree(arena_);
    arena_ = nullptr;
    if (err) *err = "cudaHostAlloc of pinned staging ring failed";
    return 0;
  }
  seg_ev_.resize(n_segs_);
  seg_ev_valid_.assign(n_segs_, false);
  for (auto& e : seg_ev_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
  batch_ev_.resize(kNBatchEv);
  fence_ev_.resize(kNBatchEv);
  for (auto& e : batch_ev_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  for (auto& e : fence_ev_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);

  slot_of_.assign(experts_.size(), -1);
  free_slots_.clear();
  free_slots_.reserve(capacity);
  for (uint32_t s = capacity; s-- > 0;) free_slots_.push_back(s);

  capacity_ = capacity;
  arena_mode_ = true;
  bytes_resident_ += (uint64_t)capacity * slot_stride_;  // arena is committed VRAM
  return capacity;
}

void DeviceModel::sync_gpu_ptr_(size_t flat, const DeviceExpert& de) {
  if (!d_ptrs_) return;
  GpuExpertPtr h;
  h.gate = (const uint8_t*)de.gate.dptr;
  h.up = (const uint8_t*)de.up.dptr;
  h.down = (const uint8_t*)de.down.dptr;
  h.resident = de.resident ? 1 : 0;
  cudaMemcpy(d_ptrs_ + flat, &h, sizeof(GpuExpertPtr), cudaMemcpyHostToDevice);
}

void DeviceModel::pin(uint32_t layer, uint32_t expert, bool value) {
  if (experts_.empty()) return;  // nothing uploaded yet
  experts_[static_cast<size_t>(layer) * n_experts_ + expert].pinned = value;
}

bool DeviceModel::evict_one_(uint64_t* bytes_freed, int64_t* freed_slot,
                             uint64_t min_age) {
  int64_t best_idx = -1;
  uint64_t best_used = 0;
  for (size_t i = 0; i < experts_.size(); ++i) {
    DeviceExpert& de = experts_[i];
    if (!de.resident || de.pinned) continue;
    if (min_age > 0 && de.last_used + min_age > tick_) continue;  // too fresh
    if (de.inflight_ev >= 0) {
      // Never reuse a slot whose speculative copy may still be writing it.
      if (cudaEventQuery(batch_ev_[de.inflight_ev]) != cudaSuccess) continue;
      de.inflight_ev = -1;  // copy done; normal candidate from here on
    }
    if (best_idx < 0 || de.last_used < best_used) {
      best_idx = (int64_t)i;
      best_used = de.last_used;
    }
  }
  if (best_idx < 0) return false;  // nothing evictable (all pinned/empty)
  DeviceExpert& de = experts_[(size_t)best_idx];
  uint64_t freed = de.gate.nbytes + de.up.nbytes + de.down.nbytes;
  if (bytes_freed) *bytes_freed += freed;
  if (arena_mode_) {
    // Slot reuse — no allocator call. Overwriting the slot is safe without a
    // sync because the reuse H2D copy is issued on the same (default) stream
    // as every kernel that could still be reading the old bytes: stream FIFO
    // orders the copy after them.
    if (freed_slot) *freed_slot = slot_of_[(size_t)best_idx];
    slot_of_[(size_t)best_idx] = -1;
  } else {
    bytes_resident_ -= freed;
    cudaFree(de.gate.dptr);
    cudaFree(de.up.dptr);
    cudaFree(de.down.dptr);
  }
  de.gate = DeviceTensor{};
  de.up = DeviceTensor{};
  de.down = DeviceTensor{};
  de.resident = false;
  sync_gpu_ptr_((size_t)best_idx, de);  // null out pointers: a stray GPU-side
                                        // dereference faults loudly instead
                                        // of silently reading evicted bytes.
  --resident_count_;
  return true;
}

void DeviceModel::stage_upload_(const ExpertBundle& b, const uint8_t* host_base,
                                DeviceExpert& de, size_t flat, int64_t slot,
                                cudaStream_t stream) {
  // mmap -> pinned segment (CPU) -> one async H2D into the slot, on `stream`
  // (nullptr = default/compute stream for reactive, copy_stream_ for prefetch).
  const uint32_t si = (uint32_t)(seg_next_ % n_segs_);
  ++seg_next_;
  if (seg_ev_valid_[si]) cudaEventSynchronize(seg_ev_[si]);
  uint8_t* seg = pin_buf_ + (uint64_t)si * slot_stride_;
  std::memcpy(seg + gate_off_, host_base + b.gate.file_offset, b.gate.nbytes);
  std::memcpy(seg + up_off_, host_base + b.up.file_offset, b.up.nbytes);
  std::memcpy(seg + down_off_, host_base + b.down.file_offset, b.down.nbytes);
  uint8_t* dst = (uint8_t*)arena_ + (uint64_t)slot * slot_stride_;
  MOEX_CUDA(cudaMemcpyAsync(dst, seg, down_off_ + b.down.nbytes,
                            cudaMemcpyHostToDevice, stream));
  MOEX_CUDA(cudaEventRecord(seg_ev_[si], stream));
  seg_ev_valid_[si] = true;

  auto mk = [&](const ByteSlice& s, uint64_t off) {
    DeviceTensor dt;
    dt.dptr = dst + off;
    dt.nbytes = s.nbytes;
    dt.type = s.type;
    dt.rows = s.rows;
    dt.cols = s.cols;
    return dt;
  };
  de.gate = mk(b.gate, gate_off_);
  de.up = mk(b.up, up_off_);
  de.down = mk(b.down, down_off_);
  slot_of_[flat] = (int32_t)slot;
}

uint32_t DeviceModel::prefetch_async(const Manifest& man, const uint8_t* host_base,
                                     uint32_t layer, const int* ids, uint32_t n,
                                     uint32_t max_uploads, uint64_t* bytes,
                                     std::string* err, uint8_t* uploaded_flags) {
  if (!arena_mode_) return 0;
  const ModelConfig& c = man.config();
  if (uploaded_flags) std::memset(uploaded_flags, 0, n);
  uint32_t issued = 0;
  const uint32_t bev = (uint32_t)(batch_next_ % kNBatchEv);
  bool fenced = false;
  ++tick_;
  try {
    for (uint32_t j = 0; j < n && issued < max_uploads; ++j) {
      int e = ids[j];
      if (e < 0 || e >= (int)c.n_experts || layer >= c.n_layers) continue;
      const size_t flat = (size_t)layer * c.n_experts + (uint32_t)e;
      DeviceExpert& de = experts_[flat];
      if (de.resident) continue;  // already there (or in flight) — free win

      int64_t slot = -1;
      if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
      } else {
        // Anti-thrash: a speculative upload may only claim a slot whose
        // occupant hasn't been routed for ~2 tokens' worth of ticks. If no
        // such victim exists the cache is fully hot — stop speculating.
        if (!evict_one_(nullptr, &slot, /*min_age=*/300)) break;
      }
      if (!fenced) {
        // One fence per batch: everything already enqueued on the compute
        // stream (which may read victim slots) must finish before the copy
        // stream writes any reused slot.
        MOEX_CUDA(cudaEventRecord(fence_ev_[bev], nullptr));
        MOEX_CUDA(cudaStreamWaitEvent(copy_stream_, fence_ev_[bev], 0));
        fenced = true;
      }
      const ExpertBundle& b = man.expert(layer, (uint32_t)e);
      stage_upload_(b, host_base, de, flat, slot, copy_stream_);
      de.resident = true;
      de.pinned = false;
      de.last_used = tick_;
      de.inflight_ev = (int)bev;
      sync_gpu_ptr_(flat, de);  // NOTE: correct only for the reactive-hit path
                                // and the gpu_dispatch fast path when used as
                                // documented (no in-flight prefetch during the
                                // fast-path window) — see Forward::gpu_dispatch.
      ++resident_count_;
      ++issued;
      if (bytes) *bytes += b.total_bytes();
      if (uploaded_flags) uploaded_flags[j] = 1;
    }
    if (issued > 0) {
      MOEX_CUDA(cudaEventRecord(batch_ev_[bev], copy_stream_));
      ++batch_next_;
    }
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
  }
  return issued;
}

uint32_t DeviceModel::ensure_bounded(const gguf::Model& gguf, const Manifest& man,
                                     const uint8_t* host_base, uint32_t layer,
                                     const int* ids, uint32_t n, std::string* err,
                                     uint64_t* bytes_uploaded,
                                     uint64_t* bytes_evicted,
                                     uint32_t* n_uploaded) {
  (void)gguf;
  const ModelConfig& c = man.config();
  if (experts_.empty()) experts_.resize((size_t)c.n_layers * c.n_experts);
  uint32_t evicted = 0;
  ++tick_;  // one recency tick per call: all ids here are "used now" together
  try {
    for (uint32_t j = 0; j < n; ++j) {
      int e = ids[j];
      if (e < 0) continue;
      const size_t flat = (size_t)layer * c.n_experts + (uint32_t)e;
      DeviceExpert& de = experts_[flat];
      de.last_used = tick_;
      if (de.resident) {
        // Hit. If a speculative copy for it may still be in flight, make the
        // compute stream wait on its batch event (device-side; host doesn't
        // block) before any kernel dereferences the slot.
        if (de.inflight_ev >= 0) {
          cudaStreamWaitEvent(nullptr, batch_ev_[de.inflight_ev], 0);
          de.inflight_ev = -1;
        }
        continue;
      }

      int64_t slot = -1;
      if (arena_mode_ && !free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
      } else if (capacity_ > 0 && resident_count_ >= capacity_) {
        if (!evict_one_(bytes_evicted, &slot)) {
          if (err) *err = "ensure_bounded: arena full, nothing unpinned to evict";
          continue;  // can't make room; leave this expert as a miss
        }
        ++evicted;
      }

      const ExpertBundle& b = man.expert(layer, (uint32_t)e);
      if (arena_mode_) {
        stage_upload_(b, host_base, de, flat, slot, /*stream=*/nullptr);
      } else {
        de.gate = upload_slice_untracked_(b.gate, host_base);
        de.up = upload_slice_untracked_(b.up, host_base);
        de.down = upload_slice_untracked_(b.down, host_base);
      }
      de.resident = true;
      sync_gpu_ptr_(flat, de);
      ++resident_count_;
      if (bytes_uploaded) *bytes_uploaded += b.total_bytes();
      if (n_uploaded) *n_uploaded += 1;
    }
  } catch (const std::exception& ex) {
    if (err) *err = ex.what();
  }
  return evicted;
}

}  // namespace moex
