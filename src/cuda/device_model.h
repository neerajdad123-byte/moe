// MoEx clean-room. Device (VRAM) residency of model weights.
//
// Uploads the static-resident tensors (embeddings, per-layer attention + norms
// + router) and a chosen working set of expert bundles from the mapped GGUF
// file straight into VRAM via cudaMemcpy. Canonical quantized bytes are kept
// AS-IS in VRAM (no FP16 expansion) — the kernels dequantize inline, matching
// design doc B2. This is the module that turns "0% VRAM" into real residency.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "gguf/gguf.h"
#include "model/manifest.h"

namespace moex {

// GPU-resident mirror of one expert's {gate,up,down,resident}. Defined here
// (not in gemv.cuh, which holds the __global__ kernels that consume it) so
// device_model.cu doesn't have to include a kernel header: nvcc's non-rdc
// compilation gives each .cu its own copy of any __global__ function a
// header pulls in, and linking two .cu's that both got a copy of the same
// kernel is a duplicate-symbol error at the host link stage.
struct GpuExpertPtr {
  const uint8_t* gate;
  const uint8_t* up;
  const uint8_t* down;
  int resident;
};

}  // namespace moex

namespace moex {

// A tensor resident in VRAM: raw quantized bytes + shape/type so a kernel can
// decode it. `dptr` is a device pointer.
struct DeviceTensor {
  void* dptr = nullptr;
  uint64_t nbytes = 0;
  gguf::GgmlType type = gguf::GgmlType::UNKNOWN;
  uint64_t rows = 0;  // logical [rows x cols], row-major over cols
  uint64_t cols = 0;
};

// One expert's three resident matrices.
struct DeviceExpert {
  DeviceTensor gate;  // [d_ff x d_model]
  DeviceTensor up;    // [d_ff x d_model]
  DeviceTensor down;  // [d_model x d_ff]
  bool resident = false;
  bool pinned = false;    // protected from ensure_bounded()'s LRU eviction
  uint64_t last_used = 0;  // tick_ snapshot, for LRU victim selection
  // >= 0 while a speculative copy-stream upload may still be in flight:
  // index into the batch-event ring. Cleared when a consumer attaches (via
  // cudaStreamWaitEvent) or eviction confirms the copy completed.
  int inflight_ev = -1;
};

// Per-layer resident static tensors.
struct DeviceLayer {
  DeviceTensor attn_norm;
  DeviceTensor attn_q, attn_k, attn_v, attn_output;
  DeviceTensor attn_q_norm, attn_k_norm;
  DeviceTensor ffn_norm;
  DeviceTensor router;  // ffn_gate_inp, F32 [n_experts x d_model]
};

class DeviceModel {
 public:
  ~DeviceModel();

  // Upload static weights for all layers + global tensors. Returns bytes used.
  // `host_base` is the mapped file base pointer (Model::mapped_data()).
  uint64_t upload_static(const gguf::Model& gguf, const Manifest& man,
                         const uint8_t* host_base, std::string* err);

  // Upload experts [layer][0..n_experts) for the layers/experts requested.
  // `which` is a per-layer list of expert ids to make resident. Returns bytes.
  // Grow-only: never evicts. Used by the unbounded resident-ceiling tools.
  uint64_t upload_experts(const gguf::Model& gguf, const Manifest& man,
                          const uint8_t* host_base,
                          const std::vector<std::vector<uint32_t>>& which,
                          std::string* err);

  // --- Bounded, evicting arena (Step 2 / hit-rate data collection) ---

  // Set the max number of experts allowed resident at once. Call once before
  // the first ensure_bounded(); 0 (default) means unbounded (evict never).
  void set_capacity(uint32_t n_experts_capacity) { capacity_ = n_experts_capacity; }

  // Slot-pool mode (design doc P0 "preallocated arena"): ONE cudaMalloc for
  // `capacity` fixed-stride expert slots + a pinned host staging ring, made
  // BEFORE the decode loop. ensure_bounded() then uploads misses via
  //   mmap -> pinned segment (CPU memcpy) -> cudaMemcpyAsync -> slot
  // on the default stream (ordered before the expert kernels by stream FIFO,
  // so the host never blocks on a copy), and eviction is just slot reuse —
  // zero cudaMalloc/cudaFree/cudaDeviceSynchronize in the loop.
  // `capacity==0` auto-sizes from live free VRAM minus `vram_reserve_bytes`.
  // Returns the chosen capacity, or 0 on failure.
  uint32_t init_arena(const Manifest& man, uint32_t capacity,
                      uint64_t vram_reserve_bytes, std::string* err);

  uint64_t slot_stride() const { return slot_stride_; }

  // Protect/unprotect one expert from ensure_bounded()'s LRU eviction.
  void pin(uint32_t layer, uint32_t expert, bool value);

  // Touch (for LRU recency) every id in `ids`, and make resident any that
  // are missing, evicting the least-recently-used *unpinned* resident expert
  // first if at capacity. Returns the number of experts evicted this call.
  // `bytes_uploaded`/`bytes_evicted`/`n_uploaded`, if given, accumulate
  // (added, not overwritten) so callers can sum precise costs across many
  // calls without diffing bytes_resident() (which can move either direction
  // per call). Unlike upload_experts, this never leaves the arena over
  // capacity. In slot-pool mode (init_arena) the upload path is
  // pinned+async and eviction never touches the CUDA allocator.
  uint32_t ensure_bounded(const gguf::Model& gguf, const Manifest& man,
                          const uint8_t* host_base, uint32_t layer,
                          const int* ids, uint32_t n, std::string* err,
                          uint64_t* bytes_uploaded = nullptr,
                          uint64_t* bytes_evicted = nullptr,
                          uint32_t* n_uploaded = nullptr);

  // Speculative prefetch (slot-pool mode only): upload up to `max_uploads`
  // missing experts of `layer` on the internal COPY stream so the DMA engine
  // overlaps compute on the default stream. Correctness:
  //  * before any slot is (re)written, the copy stream waits on a fence
  //    event recorded on the default stream NOW — so kernels already
  //    enqueued that might read a victim slot finish first;
  //  * a later ensure_bounded() that routes to a still-in-flight expert
  //    attaches a device-side cudaStreamWaitEvent instead of re-uploading;
  //  * eviction refuses experts whose in-flight copy hasn't completed.
  // Returns number of uploads issued; adds bytes to *bytes if given.
  // `uploaded_flags`, if given (size n), is set to 1 for ids actually
  // uploaded by this call (0 = was already resident / skipped / no room),
  // so callers can measure speculative precision.
  uint32_t prefetch_async(const Manifest& man, const uint8_t* host_base,
                          uint32_t layer, const int* ids, uint32_t n,
                          uint32_t max_uploads, uint64_t* bytes,
                          std::string* err, uint8_t* uploaded_flags = nullptr);

  const DeviceLayer& layer(uint32_t l) const { return layers_[l]; }
  const DeviceExpert& expert(uint32_t l, uint32_t e) const {
    return experts_[static_cast<size_t>(l) * n_experts_ + e];
  }
  const DeviceTensor& token_embd() const { return token_embd_; }
  const DeviceTensor& output_norm() const { return output_norm_; }
  const DeviceTensor& output() const { return output_; }

  uint64_t bytes_resident() const { return bytes_resident_; }
  uint32_t resident_expert_count() const { return resident_count_; }

  // GPU-resident mirror of the expert pointer table (device_model.cu keeps
  // this in sync on every residency change), for dispatch_gather_kernel —
  // the router->dispatch chain never needs the host to touch this.
  const GpuExpertPtr* device_ptr_table() const { return d_ptrs_; }

 private:
  // Allocate + copy one tensor's bytes to VRAM.
  DeviceTensor upload_tensor_(const gguf::TensorInfo* t, const uint8_t* host_base,
                              uint64_t rows, uint64_t cols);
  DeviceTensor upload_slice_(const ByteSlice& s, const uint8_t* host_base);
  // Same as upload_slice_ but does NOT register the pointer in `allocs_`:
  // used for expert bundles under ensure_bounded(), whose dptrs are freed
  // directly on eviction (and, for whatever's still resident, at teardown)
  // instead of via the flat cleanup list, to avoid a double free.
  DeviceTensor upload_slice_untracked_(const ByteSlice& s, const uint8_t* host_base);
  // Evict the LRU unpinned resident expert. Returns false if none evictable.
  // In slot-pool mode `freed_slot` receives the vacated slot index and no
  // CUDA allocator call is made; otherwise the dptrs are cudaFree'd.
  // `min_age` > 0 additionally requires the victim to be at least that many
  // ticks stale — speculative callers use it so prefetch can never thrash
  // out an expert the live route touched in the last couple of tokens.
  bool evict_one_(uint64_t* bytes_freed, int64_t* freed_slot = nullptr,
                  uint64_t min_age = 0);

  std::vector<DeviceLayer> layers_;
  std::vector<DeviceExpert> experts_;
  uint32_t n_experts_ = 0;
  DeviceTensor token_embd_, output_norm_, output_;
  std::vector<void*> allocs_;  // static-weight cudaMalloc'd ptrs, for cleanup
  uint64_t bytes_resident_ = 0;

  uint32_t capacity_ = 0;        // 0 = unbounded
  uint32_t resident_count_ = 0;  // experts currently resident
  uint64_t tick_ = 0;            // monotonic recency counter
  GpuExpertPtr* d_ptrs_ = nullptr;  // GPU mirror, size n_layers*n_experts

  // --- slot-pool arena state (only set when init_arena succeeded) ---
  bool arena_mode_ = false;
  void* arena_ = nullptr;            // one device allocation, capacity_ slots
  uint64_t slot_stride_ = 0;         // aligned max bundle bytes per slot
  uint64_t gate_off_ = 0, up_off_ = 0, down_off_ = 0;  // fixed intra-slot offsets
  std::vector<int32_t> slot_of_;     // (layer*n_experts+e) -> slot idx or -1
  std::vector<uint32_t> free_slots_; // vacated / never-used slot indices
  // pinned staging ring
  uint8_t* pin_buf_ = nullptr;
  uint32_t n_segs_ = 0;
  uint64_t seg_next_ = 0;            // monotonic; segment = seg_next_ % n_segs_
  std::vector<cudaEvent_t> seg_ev_;
  std::vector<bool> seg_ev_valid_;
  // speculative copy stream + batch/fence event rings
  cudaStream_t copy_stream_ = nullptr;
  static constexpr uint32_t kNBatchEv = 64;
  std::vector<cudaEvent_t> batch_ev_;   // recorded on copy stream per batch
  std::vector<cudaEvent_t> fence_ev_;   // recorded on default stream per batch
  uint64_t batch_next_ = 0;
  // one shared upload helper for both reactive (stream 0) and prefetch paths
  void stage_upload_(const ExpertBundle& b, const uint8_t* host_base,
                     DeviceExpert& de, size_t flat, int64_t slot,
                     cudaStream_t stream);
  // Push one entry's current {gate,up,down,resident} to the GPU table.
  // Synchronous (blocking) — this only runs on residency changes (cache
  // misses / evictions), never per-token, so a 32-byte blocking copy is
  // negligible; async would be unsafe here without a persistent staging
  // buffer since the source would be a stack temporary.
  void sync_gpu_ptr_(size_t flat, const DeviceExpert& de);
};

}  // namespace moex
