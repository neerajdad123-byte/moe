// MoEx clean-room. Forward pass implementation.
#include "cuda/forward.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cuda/cuda_common.h"
#include "cuda/gemv.cuh"
#include "cuda/kernels.cuh"

namespace moex {

// ---- element-wise / reduction kernels --------------------------------------

// RMSNorm: y = x / sqrt(mean(x^2)+eps) * w. Single block, d_model <= 4096.
__global__ void rmsnorm_kernel(const float* x, const float* w, float* y,
                               int n, float eps) {
  extern __shared__ float sh[];
  int t = threadIdx.x;
  float local = 0.0f;
  for (int i = t; i < n; i += blockDim.x) local += x[i] * x[i];
  sh[t] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  const float scale = rsqrtf(sh[0] / n + eps);
  for (int i = t; i < n; i += blockDim.x) y[i] = x[i] * scale * w[i];
}

// Per-head RMSNorm over head_dim (Qwen3 q/k norm), applied in place to vec of
// [n_heads * head_dim]. One block per head.
__global__ void head_rmsnorm_kernel(float* v, const float* w, int head_dim,
                                    float eps) {
  extern __shared__ float sh[];
  int head = blockIdx.x;
  int t = threadIdx.x;
  float* h = v + head * head_dim;
  float local = 0.0f;
  for (int i = t; i < head_dim; i += blockDim.x) local += h[i] * h[i];
  sh[t] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  const float scale = rsqrtf(sh[0] / head_dim + eps);
  for (int i = t; i < head_dim; i += blockDim.x) h[i] = h[i] * scale * w[i];
}

// Fuse per-head RMSNorm directly into RoPE. Q and K use independent
// launches, exactly like before, but remove the intermediate kernel boundary.
__global__ void head_rmsnorm_rope_kernel(float* v, const float* w, int head_dim,
                                         int pos, float eps, float base) {
  extern __shared__ float sh[];
  const int head = blockIdx.x;
  const int t = threadIdx.x;
  float* h = v + head * head_dim;
  float local = 0.0f;
  for (int i = t; i < head_dim; i += blockDim.x) local += h[i] * h[i];
  sh[t] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) sh[t] += sh[t + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(sh[0] / head_dim + eps);
  for (int i = t; i < head_dim; i += blockDim.x) h[i] *= scale * w[i];
  __syncthreads();
  const int half = head_dim / 2;
  if (t >= half) return;
  const float freq = powf(base, -2.0f * (float)t / (float)head_dim);
  const float ang = pos * freq;
  const float c = cosf(ang), ss = sinf(ang);
  const float a = h[t], b = h[t + half];
  h[t] = a * c - b * ss;
  h[t + half] = a * ss + b * c;
}

// RoPE (NEOX/GPT-style split-half rotation as used by Qwen3). Applies to a
// vector of n_heads*head_dim at absolute position pos.
__global__ void rope_kernel(float* v, int n_heads, int head_dim, int pos,
                            float base) {
  int head = blockIdx.x;
  int i = threadIdx.x;                 // 0 .. head_dim/2-1
  int half = head_dim / 2;
  if (i >= half) return;
  float* h = v + head * head_dim;
  float freq = powf(base, -2.0f * (float)i / (float)head_dim);
  float ang = pos * freq;
  float c = cosf(ang), s = sinf(ang);
  float a = h[i];
  float b = h[i + half];
  h[i] = a * c - b * s;
  h[i + half] = a * s + b * c;
}

// Copy K,V for this token into the layer cache at slot `pos`.
__global__ void kv_store_kernel(const float* k, const float* v, float* kcache,
                                float* vcache, int kv_dim, int pos) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kv_dim) return;
  kcache[(size_t)pos * kv_dim + i] = k[i];
  vcache[(size_t)pos * kv_dim + i] = v[i];
}

// GQA decode attention. One block per query head, 32 threads (one warp).
// Online softmax over timesteps with warp-shuffled m/l/acc — no atomics, no
// per-thread acc[128], no block-wide tree sync per timestep.
__global__ void gqa_attention_kernel(const float* q, const float* kcache,
                                     const float* vcache, float* out,
                                     int n_head, int n_head_kv, int head_dim,
                                     int pos, float scale) {
  const int qh = blockIdx.x;
  const int lane = threadIdx.x;  // 0..31
  const int half_ratio = n_head / n_head_kv;
  const int kvh = qh / half_ratio;
  const int kv_dim = n_head_kv * head_dim;

  extern __shared__ float qs[];
  for (int i = lane; i < head_dim; i += 32) qs[i] = q[qh * head_dim + i];
  __syncwarp();

  // Each lane owns head_dim/32 output dims (4 when head_dim=128).
  const int nvec = head_dim >> 5;  // 4
  float acc[8];                    // head_dim/32 <= 8 for head_dim<=256
  float m = -1e30f, l = 0.0f;
#pragma unroll
  for (int v = 0; v < 8; ++v) acc[v] = 0.0f;

  for (int ts = 0; ts <= pos; ++ts) {
    const float* kk = kcache + (size_t)ts * kv_dim + kvh * head_dim;
    float partial = 0.0f;
#pragma unroll
    for (int v = 0; v < nvec; ++v) partial += qs[lane + v * 32] * kk[lane + v * 32];
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, offset);
    const float score = partial * scale;

    const float mn = fmaxf(m, score);
    const float corr = expf(m - mn);
    const float p = expf(score - mn);
    l = l * corr + p;
#pragma unroll
    for (int v = 0; v < nvec; ++v) {
      const float* vv = vcache + (size_t)ts * kv_dim + kvh * head_dim;
      acc[v] = acc[v] * corr + p * vv[lane + v * 32];
    }
    m = mn;
  }

  const float inv = 1.0f / l;
#pragma unroll
  for (int v = 0; v < nvec; ++v)
    out[qh * head_dim + lane + v * 32] = acc[v] * inv;
}

// SiLU(gate) * up  ->  gate_buf
__global__ void silu_mul_kernel(float* gate, const float* up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

// x += scale * y  (residual with per-expert weight)
__global__ void axpy_kernel(float* x, const float* y, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += scale * y[i];
}

// plain add residual: x += y
__global__ void add_kernel(float* x, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] += y[i];
}

// embedding lookup: dequantize row `token` of token_embd into x[d_model].
__global__ void embed_kernel(int type, const uint8_t* W, int token, float* x,
                             int d_model, int row_bytes) {
  // one warp dequantizes the row cooperatively (reuse row_dot-like decode).
  int lane = threadIdx.x & 31;
  const uint8_t* row = W + (size_t)token * row_bytes;
  // Each lane decodes superblocks and writes directly.
  if (type == GT_Q4_K || type == GT_Q6_K || type == GT_Q5_K) {
    int nsb = d_model / 256;
    int blk = (type == GT_Q4_K) ? BQ4_K : (type == GT_Q5_K) ? BQ5_K : BQ6_K;
    float buf[256];
    for (int sb = lane; sb < nsb; sb += 32) {
      const uint8_t* p = row + (size_t)sb * blk;
      if (type == GT_Q4_K) deq_superblock_q4k(p, buf);
      else if (type == GT_Q5_K) deq_superblock_q5k(p, buf);
      else deq_superblock_q6k(p, buf);
      for (int i = 0; i < 256; ++i) x[sb * 256 + i] = buf[i];
    }
  }
}

// ---- debug stats probe ------------------------------------------------------
// Reduce a device buffer to {min,max,sum,nan,inf} so we can see exactly where
// the forward pass first produces garbage. Gated by MOEX_DEBUG env var.
__global__ void stats_kernel(const float* v, int n, float* out) {
  // out: [min,max,sum,nancount,infcount]; single block, atomic into out.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) { out[0] = 1e30f; out[1] = -1e30f; out[2]=0; out[3]=0; out[4]=0; }
  __syncthreads();
  if (i >= n) return;
  float x = v[i];
  if (isnan(x)) { atomicAdd(&out[3], 1.0f); return; }
  if (isinf(x)) { atomicAdd(&out[4], 1.0f); return; }
  atomicMin((int*)&out[0], __float_as_int(x) ^ ((__float_as_int(x) >> 31) & 0x7fffffff));
  atomicAdd(&out[2], x);
}

static bool debug_on() {
  static int v = -1;
  if (v < 0) { const char* e = getenv("MOEX_DEBUG"); v = (e && e[0]=='1') ? 1 : 0; }
  return v == 1;
}

// Non-finite detection by IEEE-754 bit pattern: exp==0xFF means inf/nan,
// regardless of any fast-math flag that might defeat std::isinf/isnan.
static inline bool bits_nonfinite(float x, bool* is_nan) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  uint32_t exp = (u >> 23) & 0xFF;
  uint32_t man = u & 0x7FFFFF;
  if (exp == 0xFF) { *is_nan = (man != 0); return true; }
  *is_nan = false;
  return false;
}

static void dbg(const char* tag, const float* dv, int n) {
  if (!debug_on()) return;
  std::vector<float> h(n);
  cudaMemcpy(h.data(), dv, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
  double mn = 1e30, mx = -1e30, sum = 0; int nan = 0, inf = 0; int fin = 0;
  for (int i = 0; i < n; ++i) {
    float x = h[i];
    bool isnan;
    if (bits_nonfinite(x, &isnan)) { if (isnan) ++nan; else ++inf; continue; }
    mn = x < mn ? x : mn; mx = x > mx ? x : mx; sum += x; ++fin;
  }
  std::fprintf(stderr,
               "  [dbg] %-16s n=%-6d fin=%-6d nan=%d inf=%d min=%+.4e max=%+.4e mean=%+.4e | [0..4]= %+.4e %+.4e %+.4e %+.4e %+.4e\n",
               tag, n, fin, nan, inf, mn, mx, sum / (fin > 0 ? fin : 1),
               h[0], n>1?h[1]:0, n>2?h[2]:0, n>3?h[3]:0, n>4?h[4]:0);
}

// ---- host orchestration ----------------------------------------------------

static inline int gt(gguf::GgmlType t) { return (int)t; }

Forward::Forward(const DeviceModel& dm, const ModelConfig& cfg, uint32_t max_ctx,
                 const Manifest& man)
    : dm_(dm), cfg_(cfg), max_ctx_(max_ctx) {
  auto alloc = [](float** p, size_t n) {
    MOEX_CUDA(cudaMalloc(p, n * sizeof(float)));
  };
  alloc(&x_, cfg.d_model);
  alloc(&xn_, cfg.d_model);
  alloc(&q_, cfg.q_dim());
  alloc(&k_, cfg.kv_dim());
  alloc(&v_, cfg.kv_dim());
  alloc(&attn_out_, cfg.q_dim());
  alloc(&tmp_, cfg.d_model);
  alloc(&router_logits_, cfg.n_experts);
  alloc(&gate_buf_, cfg.d_ff_expert);
  alloc(&up_buf_, cfg.d_ff_expert);
  alloc(&expert_out_, cfg.d_model);
  alloc(&group_gate_buf_, (size_t)cfg.n_experts_used * cfg.d_ff_expert);
  alloc(&group_expert_out_, (size_t)cfg.n_experts_used * cfg.d_model);
  MOEX_CUDA(cudaMalloc(&d_dispatch_, sizeof(ExpertDispatch)));
  MOEX_CUDA(cudaMallocHost(&h_dispatch_, sizeof(ExpertDispatch)));
  alloc(&logits_, cfg.vocab_size);
  alloc(&kcache_, (size_t)cfg.n_layers * max_ctx * cfg.kv_dim());
  alloc(&vcache_, (size_t)cfg.n_layers * max_ctx * cfg.kv_dim());
  alloc(&argmax_part_v_, argmax_nblocks_);
  MOEX_CUDA(cudaMalloc(&argmax_part_i_, argmax_nblocks_ * sizeof(int)));
  MOEX_CUDA(cudaMalloc(&d_argmax_, sizeof(int)));
  MOEX_CUDA(cudaMalloc(&d_route_, sizeof(RouteTopK)));
  MOEX_CUDA(cudaMallocHost(&h_route_, sizeof(RouteTopK)));
  MOEX_CUDA(cudaMalloc(&d_gdispatch_, sizeof(ExpertDispatch)));
  MOEX_CUDA(cudaMalloc(&d_hit_ctr_, sizeof(unsigned long long)));
  MOEX_CUDA(cudaMalloc(&d_miss_ctr_, sizeof(unsigned long long)));
  MOEX_CUDA(cudaMemset(d_hit_ctr_, 0, sizeof(unsigned long long)));
  MOEX_CUDA(cudaMemset(d_miss_ctr_, 0, sizeof(unsigned long long)));
  h_logits_ = (float*)malloc((size_t)cfg.vocab_size * sizeof(float));
  h_router_ = (float*)malloc((size_t)cfg.n_experts * sizeof(float));
  last_route_.assign((size_t)cfg.n_layers * cfg.n_experts_used, -1);

  // Per-layer expert tensor type/row-bytes, from the manifest (not from any
  // resident DeviceExpert — geometry is uniform across a layer's 128 experts,
  // and the gpu_dispatch fast path never learns a specific expert id on the
  // host to key a DeviceExpert lookup off of).
  gu_type_by_layer_.resize(cfg.n_layers);
  dn_type_by_layer_.resize(cfg.n_layers);
  gu_rowbytes_by_layer_.resize(cfg.n_layers);
  dn_rowbytes_by_layer_.resize(cfg.n_layers);
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const ExpertBundle& b = man.expert(l, 0);
    gu_type_by_layer_[l] = (int)b.gate.type;
    dn_type_by_layer_[l] = (int)b.down.type;
    gu_rowbytes_by_layer_[l] = (int)(b.gate.nbytes / cfg.d_ff_expert);
    dn_rowbytes_by_layer_[l] = (int)(b.down.nbytes / cfg.d_model);
  }
}

Forward::~Forward() {
  for (float* p : {x_, xn_, q_, k_, v_, attn_out_, tmp_, router_logits_,
                   gate_buf_, up_buf_, expert_out_, group_gate_buf_,
                   group_expert_out_, logits_, kcache_, vcache_}) {
    if (p) cudaFree(p);
  }
  for (void* p : {(void*)argmax_part_v_, (void*)argmax_part_i_,
                  (void*)d_argmax_, (void*)d_route_, (void*)d_gdispatch_,
                  (void*)d_hit_ctr_, (void*)d_miss_ctr_}) {
    if (p) cudaFree(p);
  }
  if (h_route_) cudaFreeHost(h_route_);
  if (d_dispatch_) cudaFree(d_dispatch_);
  if (h_dispatch_) cudaFreeHost(h_dispatch_);
  free(h_logits_);
  free(h_router_);
}

void Forward::read_hit_miss_counters(unsigned long long* hits,
                                     unsigned long long* misses) const {
  if (hits) MOEX_CUDA(cudaMemcpy(hits, d_hit_ctr_, sizeof(unsigned long long),
                                 cudaMemcpyDeviceToHost));
  if (misses) MOEX_CUDA(cudaMemcpy(misses, d_miss_ctr_, sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost));
}

void Forward::reset_hit_miss_counters() {
  MOEX_CUDA(cudaMemset(d_hit_ctr_, 0, sizeof(unsigned long long)));
  MOEX_CUDA(cudaMemset(d_miss_ctr_, 0, sizeof(unsigned long long)));
}

int Forward::step(int token_id, int pos, std::vector<int>* route_out,
                  std::vector<float>* weight_out) {
  const ModelConfig& c = cfg_;
  const int D = c.d_model;
  const int threads = 256;
  auto blocks = [](int n, int t) { return (n + t - 1) / t; };
  const float attn_scale = 1.0f / sqrtf((float)c.head_dim);

  // Optional per-phase timing. In profiled mode we sync+time coarse phases on
  // the host; this serializes and perturbs absolute latency, so the HEADLINE
  // tok/s always comes from a separate run with prof==nullptr (two-run
  // discipline). Relative phase breakdown is what the profiled run reports.
  StepProf* pf = prof;
  auto hit = [&](const char* name, int n = 1) {
    if (pf) pf->launches.hit(name, n);
  };
  auto phase = [&](double* acc) {
    if (!pf) return;
    cudaDeviceSynchronize();
    *acc += ms_since(pf->tp);
    pf->tp = Clock::now();
  };
  // Analytical dequant accounting: quantized weight bytes read for one GEMV
  // (fused dequant inside row_dot — no separate dequant kernel/cache).
  auto account_gemv = [&](uint64_t nbytes, int out_rows, int in_cols) {
    if (!pf) return;
    pf->dequant_bytes_in += nbytes;
    pf->dequant_elems_out += (uint64_t)out_rows * (uint64_t)in_cols;
  };
  auto tok_t0 = Clock::now();
  const uint64_t h2d_before = pf ? pf->h2d_bytes : 0;
  const uint64_t d2h_before = pf ? pf->d2h_bytes : 0;
  const uint64_t d2d_before = pf ? pf->d2d_bytes : 0;
  const double embed_before = pf ? pf->embed_ms : 0.0;
  const double attn_before = pf ? pf->attn_ms : 0.0;
  const double router_before = pf ? pf->router_ms : 0.0;
  const double experts_before = pf ? pf->experts_ms : 0.0;
  const double final_before = pf ? pf->final_ms : 0.0;
  int ev0 = -1, ev1 = -1, ev2 = -1, ev3 = -1, ev4 = -1, ev5 = -1;
  if (pf) {
    if (pf->expert_hist.size() != (size_t)c.n_layers * c.n_experts)
      pf->expert_hist.assign((size_t)c.n_layers * c.n_experts, 0);
    cudaDeviceSynchronize();
    pf->tp = Clock::now();
    pf->pool.reset();
    ev0 = pf->pool.mark();
  }

  // Embedding: x = token_embd[token]
  embed_kernel<<<1, 32>>>(gt(dm_.token_embd().type),
                          (const uint8_t*)dm_.token_embd().dptr, token_id, x_,
                          D, (int)(dm_.token_embd().nbytes / c.vocab_size));
  hit("embed");
  account_gemv(dm_.token_embd().nbytes / c.vocab_size, 1, D);
  dbg("embed", x_, D);
  if (pf) { ev1 = pf->pool.mark(); phase(&pf->embed_ms); }

  for (uint32_t l = 0; l < c.n_layers; ++l) {
    const DeviceLayer& dl = dm_.layer(l);
    const bool dbgl = (l == 0);
    const bool win = debug_on() && (l == 25);  // NaN-birth window
    auto wprobe = [&](const char* what, const float* buf, int n) {
      if (!win) return;
      char tg[40]; std::snprintf(tg, sizeof tg, "l%u.%s", l, what);
      dbg(tg, buf, n);
    };

    // --- attention block ---
    wprobe("in.x", x_, D);
    rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(
        x_, (const float*)dl.attn_norm.dptr, xn_, D, c.rms_eps);
    hit("rmsnorm");

    if (dbgl) dbg("l0.attn_norm", xn_, D);
    wprobe("attn_norm", xn_, D);

    // One launch for Q/K/V while preserving their distinct GGUF layouts:
    // Q4_K (Q), Q8_0 (K), and Q6_K (V) in this model.
    launch_qkv_q(
        gt(dl.attn_q.type), (const uint8_t*)dl.attn_q.dptr,
        (int)(dl.attn_q.nbytes / c.q_dim()), gt(dl.attn_k.type),
        (const uint8_t*)dl.attn_k.dptr, (int)(dl.attn_k.nbytes / c.kv_dim()),
        gt(dl.attn_v.type), (const uint8_t*)dl.attn_v.dptr,
        (int)(dl.attn_v.nbytes / c.kv_dim()), xn_, q_, k_, v_, c.q_dim(),
        c.kv_dim(), D);
    hit("qkv_fused");
    account_gemv(dl.attn_q.nbytes, c.q_dim(), D);
    account_gemv(dl.attn_k.nbytes, c.kv_dim(), D);
    account_gemv(dl.attn_v.nbytes, c.kv_dim(), D);

    // Fuse per-head q/k RMSNorm with RoPE.
    head_rmsnorm_rope_kernel<<<c.n_head, 128, 128 * sizeof(float)>>>(
        q_, (const float*)dl.attn_q_norm.dptr, c.head_dim, pos, c.rms_eps,
        c.rope_base);
    hit("head_rmsnorm_rope");
    head_rmsnorm_rope_kernel<<<c.n_head_kv, 128, 128 * sizeof(float)>>>(
        k_, (const float*)dl.attn_k_norm.dptr, c.head_dim, pos, c.rms_eps,
        c.rope_base);
    hit("head_rmsnorm_rope");

    // Store K/V into cache.
    float* kc = kcache_ + (size_t)l * kv_stride_layer_();
    float* vc = vcache_ + (size_t)l * kv_stride_layer_();
    kv_store_kernel<<<blocks(c.kv_dim(), threads), threads>>>(
        k_, v_, kc, vc, c.kv_dim(), pos);
    hit("kv_store");
    if (pf) {
      pf->d2d_bytes += 2ull * c.kv_dim() * sizeof(float);
      pf->d2d_calls += 1;
    }

    if (dbgl) dbg("l0.q(after rope)", q_, c.q_dim());
    wprobe("q_postrope", q_, c.q_dim());
    wprobe("k_postrope", k_, c.kv_dim());

    // Attention: one warp per head (shuffle reduce, no atomics).
    gqa_attention_kernel<<<c.n_head, 32, (size_t)c.head_dim * sizeof(float)>>>(
        q_, kc, vc, attn_out_, c.n_head, c.n_head_kv, c.head_dim, pos,
        attn_scale);
    hit("gqa_attn");
    if (dbgl) dbg("l0.attn_out", attn_out_, c.q_dim());
    wprobe("attn_out", attn_out_, c.q_dim());

    // Fused output projection plus residual.
    launch_gemv_q_residual(
        gt(dl.attn_output.type), (const uint8_t*)dl.attn_output.dptr, attn_out_,
        x_, D, c.q_dim(), (int)(dl.attn_output.nbytes / D));
    hit("gemv_attn_out_residual");
    account_gemv(dl.attn_output.nbytes, D, c.q_dim());
    if (dbgl) dbg("l0.x(after attn)", x_, D);
    wprobe("x_postattn", x_, D);
    if (pf) { ev2 = pf->pool.mark(); phase(&pf->attn_ms); }

    // --- MoE block ---
    rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(
        x_, (const float*)dl.ffn_norm.dptr, xn_, D, c.rms_eps);
    hit("rmsnorm");
    wprobe("ffn_norm", xn_, D);

    // Router logits = Wr . xn  (F32 weights [n_experts x d_model])
    gemv_q<<<blocks(c.n_experts * 32, threads), threads, gemv_smem_bytes(D)>>>( 
        gt(dl.router.type), (const uint8_t*)dl.router.dptr, xn_,
        router_logits_, c.n_experts, D, (int)(dl.router.nbytes / c.n_experts));
    hit("gemv_router");
    account_gemv(dl.router.nbytes, c.n_experts, D);
    if (dbgl) dbg("l0.router_logits", router_logits_, c.n_experts);
    if (pf) ev3 = pf->pool.mark();

    // gpu_dispatch: keep the fast path even while profiling (phase syncs still
    // inflate absolute ms; headline tok/s stays on prof==nullptr). Skip only
    // when force_route needs host-visible ids.
    if (gpu_dispatch && !force_route) {
      // Zero-host-round-trip path: router top-k, pointer resolution, and
      // dispatch all stay on GPU, in ONE fused launch (see kernel doc: a
      // separate dispatch_gather_kernel measured net SLOWER on this WDDM
      // setup — the extra launch cost more than the D2H it removed). See
      // Forward::gpu_dispatch doc comment for the residency contract.
      const uint32_t layer_base = l * c.n_experts;
      router_topk8_dispatch_kernel<<<1, 128>>>(
          router_logits_, (int)c.n_experts, (int)c.n_experts_used, d_route_,
          dm_.device_ptr_table(), layer_base, d_gdispatch_, d_hit_ctr_,
          d_miss_ctr_);
      hit("router_topk8_dispatch");
      if (pf) {
        cudaDeviceSynchronize();
        pf->router_ms += ms_since(pf->tp);
        pf->tp = Clock::now();
      }
      launch_expert_group_gate_up_silu_dptr(
          gu_type_by_layer_[l], d_gdispatch_, xn_, group_gate_buf_,
          (int)c.n_experts_used, c.d_ff_expert, D, gu_rowbytes_by_layer_[l]);
      hit("expert_group_gate_up_silu_dptr");
      launch_expert_group_down_dptr(
          dn_type_by_layer_[l], d_gdispatch_, group_gate_buf_,
          group_expert_out_, (int)c.n_experts_used, D, c.d_ff_expert,
          dn_rowbytes_by_layer_[l]);
      hit("expert_group_down_dptr");
      launch_expert_group_residual_dptr(d_gdispatch_, group_expert_out_, x_,
                                        (int)c.n_experts_used, D);
      hit("expert_group_residual_dptr");
      if (pf) {
        for (uint32_t j = 0; j < c.n_experts_used; ++j) {
          account_gemv((uint64_t)gu_rowbytes_by_layer_[l] * c.d_ff_expert,
                       c.d_ff_expert, D);
          account_gemv((uint64_t)gu_rowbytes_by_layer_[l] * c.d_ff_expert,
                       c.d_ff_expert, D);
          account_gemv((uint64_t)dn_rowbytes_by_layer_[l] * D, D,
                       c.d_ff_expert);
        }
        ev4 = pf->pool.mark();
        phase(&pf->experts_ms);
      }
    } else {
    // Router softmax + top-k ON GPU; host receives one 64 B RouteTopK copy
    // (was: 512 B logits D2H + host softmax + partial_sort per layer).
    auto t_rs = Clock::now();
    std::vector<int> idx(c.n_experts_used);
    float sel_w[64];
    if (force_route) {
      // Debug/ceiling path: forced ids still need full logits for weights.
      MOEX_CUDA(cudaMemcpy(h_router_, router_logits_,
                           c.n_experts * sizeof(float), cudaMemcpyDeviceToHost));
      if (pf) {
        pf->d2h_bytes += (uint64_t)c.n_experts * sizeof(float);
        pf->d2h_calls += 1;
      }
      for (uint32_t j = 0; j < c.n_experts_used; ++j)
        idx[j] = force_route[(size_t)l * c.n_experts_used + j];
      float maxl = -1e30f;
      for (int i = 0; i < (int)c.n_experts; ++i) maxl = fmaxf(maxl, h_router_[i]);
      double denom = 0.0;
      for (int i = 0; i < (int)c.n_experts; ++i)
        denom += exp((double)(h_router_[i] - maxl));
      double sel_sum = 0.0;
      for (uint32_t j = 0; j < c.n_experts_used; ++j) {
        double p = exp((double)(h_router_[idx[j]] - maxl)) / denom;
        sel_w[j] = (float)p;
        sel_sum += p;
      }
      for (uint32_t j = 0; j < c.n_experts_used; ++j) sel_w[j] /= (float)sel_sum;
    } else {
      router_topk8_kernel<<<1, 128>>>(router_logits_, (int)c.n_experts,
                                      (int)c.n_experts_used, d_route_);
      hit("router_topk8");
      MOEX_CUDA(cudaMemcpy(h_route_, d_route_, sizeof(RouteTopK),
                           cudaMemcpyDeviceToHost));
      if (pf) {
        pf->d2h_bytes += sizeof(RouteTopK);
        pf->d2h_calls += 1;
      }
      for (uint32_t j = 0; j < c.n_experts_used; ++j) {
        idx[j] = h_route_->ids[j];
        sel_w[j] = h_route_->w[j];
      }
    }
    if (pf) {
      pf->router_sync_ms += ms_since(t_rs);
      // router_ms = GPU router gemv + host top-k window from last phase mark
      pf->router_ms += ms_since(pf->tp);
      pf->tp = Clock::now();
    }

    // Record route + let the caller make these experts resident (reactive
    // upload) before we dereference their device pointers.
    int route_ids[64];
    for (uint32_t j = 0; j < c.n_experts_used; ++j) {
      int e = idx[j];
      route_ids[j] = e;
      last_route_[(size_t)l * c.n_experts_used + j] = e;
      if (route_out) (*route_out)[(size_t)l * c.n_experts_used + j] = e;
      if (weight_out) (*weight_out)[(size_t)l * c.n_experts_used + j] = sel_w[j];
      if (pf) {
        pf->expert_hist[(size_t)l * c.n_experts + e] += 1;
        if (dm_.expert(l, e).resident) pf->expert_hits += 1;
        else pf->expert_miss += 1;
      }
    }
    // Reactive upload hook (route discovery / paging). Same-stream ordering
    // already sequences xn_ producers before the expert GEMVs, and the router
    // D2H copy above is itself blocking, so no extra device sync is needed here.
    if (ensure_experts) {
      ensure_experts(l, route_ids, c.n_experts_used);
    }

    // Grouped dispatch: all top-k experts in 3 launches (was 16), with the
    // pointer table passed by value in kernel constant params — no global
    // pointer-table reads, no host->device dispatch copy, and a full-GPU
    // grid (k x d_ff blocks) instead of k sequential small waves.
    {
      ExpertDispatch disp{};
      const DeviceExpert& de0 = dm_.expert(l, idx[0]);
      const int gu_row_bytes = (int)(de0.gate.nbytes / c.d_ff_expert);
      const int dn_row_bytes = (int)(de0.down.nbytes / D);
      const int gu_type = gt(de0.gate.type);
      const int dn_type = gt(de0.down.type);
      for (uint32_t j = 0; j < c.n_experts_used; ++j) {
        const DeviceExpert& de = dm_.expert(l, idx[j]);
        disp.gate[j] = (const uint8_t*)de.gate.dptr;
        disp.up[j] = (const uint8_t*)de.up.dptr;
        disp.down[j] = (const uint8_t*)de.down.dptr;
        disp.weight[j] = sel_w[j];
        account_gemv(de.gate.nbytes, c.d_ff_expert, D);
        account_gemv(de.up.nbytes, c.d_ff_expert, D);
        account_gemv(de.down.nbytes, D, c.d_ff_expert);
      }
      launch_expert_group_gate_up_silu(gu_type, disp, xn_, group_gate_buf_,
                                       (int)c.n_experts_used, c.d_ff_expert, D,
                                       gu_row_bytes);
      hit("expert_group_gate_up_silu");
      launch_expert_group_down(dn_type, disp, group_gate_buf_,
                               group_expert_out_, (int)c.n_experts_used, D,
                               c.d_ff_expert, dn_row_bytes);
      hit("expert_group_down");
      launch_expert_group_residual(disp, group_expert_out_, x_,
                                   (int)c.n_experts_used, D);
      hit("expert_group_residual");
    }
    if (pf) { ev4 = pf->pool.mark(); phase(&pf->experts_ms); }
    }  // else (non-gpu_dispatch path)
    if (debug_on()) {  // per-layer: catch first non-finite AND any launch error
      cudaError_t e1 = cudaDeviceSynchronize();
      cudaError_t e2 = cudaGetLastError();
      if (e1 != cudaSuccess || e2 != cudaSuccess) {
        std::fprintf(stderr, "  [dbg] *** LAUNCH/EXEC ERROR at layer %u: sync=%s last=%s\n",
                     l, cudaGetErrorString(e1), cudaGetErrorString(e2));
      }
      char tag[24]; std::snprintf(tag, sizeof(tag), "l%u.x_end", l);
      dbg(tag, x_, D);
    }
  }

  // Final norm + logits.
  dbg("prefinal.x", x_, D);
  rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(
      x_, (const float*)dm_.output_norm().dptr, xn_, D, c.rms_eps);
  hit("rmsnorm");
  dbg("final_norm", xn_, D);
  gemv_q<<<blocks(c.vocab_size * 32, threads), threads, gemv_smem_bytes(D)>>>(
      gt(dm_.output().type), (const uint8_t*)dm_.output().dptr, xn_, logits_,
      c.vocab_size, D, (int)(dm_.output().nbytes / c.vocab_size));
  hit("gemv_logits");
  account_gemv(dm_.output().nbytes, c.vocab_size, D);
  dbg("final_logits", logits_, (int)c.vocab_size);
  if (pf) ev5 = pf->pool.mark();

  // Greedy sampling ON GPU (design doc P0 "on-GPU sampling"): two-stage
  // argmax over 151,936 logits, then a 4-byte D2H — replaces a 608 KB logits
  // copy + host scan every token.
  auto t_ls = Clock::now();
  argmax_stage1_kernel<<<argmax_nblocks_, 256>>>(logits_, (int)c.vocab_size,
                                                 argmax_part_v_, argmax_part_i_);
  hit("argmax_stage1");
  argmax_stage2_kernel<<<1, 256>>>(argmax_part_v_, argmax_part_i_,
                                   argmax_nblocks_, d_argmax_);
  hit("argmax_stage2");
  int best = 0;
  MOEX_CUDA(cudaMemcpy(&best, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  if (pf) {
    pf->d2h_bytes += sizeof(int);
    pf->d2h_calls += 1;
    pf->logits_sync_ms += ms_since(t_ls);
    pf->final_ms += ms_since(pf->tp);
    pf->tp = Clock::now();
    pf->sample_vram();
  }

  if (debug_on()) {
    MOEX_CUDA(cudaMemcpy(h_logits_, logits_, c.vocab_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    std::fprintf(stderr, "  [dbg] raw logits[0..9]:");
    for (int i = 0; i < 10; ++i) std::fprintf(stderr, " %.3e", h_logits_[i]);
    std::fprintf(stderr, "\n");
  }
  if (pf) {
    const double wall_ms = ms_since(tok_t0);
    pf->tok_ms.push_back(wall_ms);
    TokenTrace tr;
    tr.pos = pos;
    tr.input_token = token_id;
    tr.wall_ms = wall_ms;
    // Whole-token synchronized phase totals (all 48 layers), not a last-layer
    // event span. Kernel-exclusive attribution is collected separately by NCU.
    tr.embed_gpu_ms = (float)(pf->embed_ms - embed_before);
    tr.attention_gpu_ms = (float)(pf->attn_ms - attn_before);
    tr.router_gpu_ms = (float)(pf->router_ms - router_before);
    tr.experts_gpu_ms = (float)(pf->experts_ms - experts_before);
    tr.final_gpu_ms = (float)(pf->final_ms - final_before);
    tr.h2d_bytes = pf->h2d_bytes - h2d_before;
    tr.d2h_bytes = pf->d2h_bytes - d2h_before;
    tr.d2d_bytes = pf->d2d_bytes - d2d_before;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      tr.vram_used = (uint64_t)(total_b - free_b);
      if (tr.vram_used > pf->vram_peak_used) pf->vram_peak_used = tr.vram_used;
    }
    pf->trace.push_back(tr);
  }
  return best;
}

}  // namespace moex
