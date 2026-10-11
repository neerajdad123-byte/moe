# MoEx v2 — build plan

Fresh CUDA engine for **Qwen3.6-35B-A3B** (`D:\models\qwen36_i4_gs64`, colibri int4 gs64 container)
on an RTX 4050 Laptop (6 GB, 188 GB/s measured, PCIe 4.0 x8), i7-13650HX, 24 GB DDR5-4800.

Goal: fastest exact batch-1 decode on this box, beating llama.cpp / ik_llama / colibri on the same model.

## Rules

1. **Reference first.** Every kernel is checked against `ref/qwen36_ref.py` (PyTorch, fp32) before it is used.
2. **A kernel is "proven" when** it (a) matches the reference within tolerance and (b) has a benchmark
   reporting achieved GB/s vs the 188 GB/s roofline. Numbers go in the status table below.
3. **Correct, then fast.** First version of each kernel is the simple one; upgrades must beat it on the bench.
4. Speed claims come only from `bench` runs, never estimates.

## Model facts (from the container)

| | |
|---|---|
| Layers | 40: 30 Gated-DeltaNet + 10 full attention (every 4th) |
| Hidden | 2048, vocab 248320 |
| MoE | 256 experts, top-8, softmax router, `norm_topk_prob=false`, + shared expert (sigmoid-gated) |
| Expert | int4 gs64 symmetric, `merged_weight` = [gate 512x2048 \| up 512x2048 \| down 2048x512], 1.77 MB |
| Dense trunk | **F16** (not quantized): 2.86 GB + lm_head 1.02 GB |
| Attention | 16 q heads / 2 kv heads, head_dim 256, q_proj also emits a sigmoid output gate, q/k RMSNorm, partial RoPE (64 of 256 dims), theta 1e7 |
| DeltaNet | 16 k-heads / 32 v-heads, dim 128, conv k=4 over 8192 ch, state 32x128x128 f32 per layer |
| RMSNorm | `x * rsqrt(mean(x^2)+eps) * (1 + w)`; DeltaNet gated norm uses plain `w` and `silu(z)` |
| MTP head | **not in this container** (converter drops it) |

Per-token bytes if everything were on GPU: dense F16 3.88 GB + experts 0.57 GB → **~48 tok/s hard ceiling with F16 dense**.
Dense int8 → ~75 tok/s ceiling; dense int4 → ~110. So dense requantization is a planned upgrade (U1).

## Build list

### Phase 0 — ground truth
- [ ] P0.1 `ref/qwen36_ref.py`: container loader + full fp32 forward (DeltaNet, gated attention, MoE). Sanity: greedy text is coherent.
- [ ] P0.2 Golden dumps: per-op inputs/outputs for one prompt (`ref/golden/*.bin`), route traces per token.

### Phase 1 — kernels (each: unit test vs golden + bench)
- [ ] K1 RMSNorm (1+w), fused residual-add variant
- [ ] K2 Dense GEMV F16 (x fp32 → y fp32)
- [ ] K3 Router: gate GEMV 256 + softmax + top-8
- [ ] K4 Expert int4 gs64: fused gate+up+SiLU·mul, then down with weighted sum over k experts
- [ ] K5 Shared expert + sigmoid gate (reuses K2)
- [ ] K6 q/k per-head RMSNorm + partial RoPE + KV-cache append
- [ ] K7 Decode attention: GQA, head_dim 256, softmax, output gate
- [ ] K8 DeltaNet decode: conv1d ring + SiLU, l2norm, gates (beta, g), recurrent state update, gated RMSNorm
- [ ] K9 lm_head GEMV + argmax (sampling later)

### Phase 2 — engine (all-GPU correctness)
- [ ] E1 Loader: mmap safetensors, RAM budget, upload trunk to VRAM
- [ ] E2 Expert arena: fixed slots in VRAM + slot table; on-demand H2D fetch on miss (simplest correct path)
- [ ] E3 Decode step wiring all kernels; logits match reference
- [ ] E4 Tokenizer (shortcut: Python `tokenizers` front-end at first)
- [ ] E5 `moex generate` CLI + per-phase profiler + tok/s

### Phase 3 — offload engine (the point of the project)
- [ ] O1 CPU expert kernel (AVX2, int4 gs64, P-cores), checked vs reference
- [ ] O2 Hybrid scheduler: resident experts on GPU, misses on CPU, overlapped per layer
- [ ] O3 Residency policy: LRU baseline → decayed-frequency (τ sweep)
- [ ] O4 Prefetch on copy stream: cross-layer Markov predictor, PCIe budgeted
- [ ] O5 Route traces for this model + port `simulate_policies.py` (old repo) to 256 experts

### Phase 4 — head-to-head
- [ ] B1 colibri CUDA tier on the same container (already on disk)
- [ ] B2 llama.cpp / ik_llama `--n-cpu-moe` on a Qwen3.6-35B-A3B GGUF (needs download)

### Upgrades (only after Phase 2 works)
- [ ] U1 Dense trunk requant F16 → int8 (then int4 gs64): frees ~2 GB VRAM for experts, halves bytes/token
- [ ] U2 dp4a / int8-activation expert GEMV
- [ ] U3 CUDA graphs per decode step
- [ ] U4 Prefill path (batched GEMM, chunked DeltaNet)
- [ ] U5 Sampling (temperature / top-p), chat template, server

## Status

| Item | Correct | Bench | Notes |
|---|---|---|---|
