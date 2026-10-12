# MoEx v2 — build plan

Fresh CUDA engine for **Qwen3.6-35B-A3B** (`C:\models\qwen36_i4_gs64`, NVMe SSD, colibri int4 gs64 container)
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
- [x] P0.1 `ref/qwen36_ref.py`: container loader + full fp32 forward (DeltaNet, gated attention, MoE). Sanity: greedy text is coherent.
- [x] P0.2 Golden dumps: per-op inputs/outputs for one prompt (`ref/golden/*.bin`), route traces per token.
  Regenerate: `python ref/qwen36_ref.py "The capital of France is" --n 8 --dump ref/golden --routes ref/golden/routes.jsonl`

### Phase 1 — kernels (each: unit test vs golden + bench)
- [x] K1 RMSNorm (1+w), fused residual-add variant
- [x] K2 Dense GEMV F16 (x fp32 → y fp32)
- [x] K3 Router: gate GEMV 256 + softmax + top-8
- [x] K4 Expert int4 gs64: fused gate+up+SiLU·mul, then down with weighted sum over k experts
- [x] K5 Shared expert + sigmoid gate (reuses K2)
- [x] K6 q/k per-head RMSNorm + partial RoPE + KV-cache append
- [x] K7 Decode attention: GQA, head_dim 256, softmax, output gate
- [x] K8 DeltaNet decode: conv1d ring + SiLU, l2norm, gates (beta, g), recurrent state update, gated RMSNorm
- [x] K9 lm_head GEMV + argmax (sampling later)

### Phase 2 — engine (all-GPU correctness)
- [x] E1 Loader: mmap safetensors, RAM budget, upload trunk to VRAM
- [x] E2 Expert arena: fixed slots in VRAM + slot table; on-demand H2D fetch on miss (simplest correct path)
- [x] E3 Decode step wiring all kernels; logits match reference
- [x] E4 Tokenizer (shortcut: Python `tokenizers` front-end at first)
- [x] E5 `moex generate` CLI + per-phase profiler + tok/s

### Phase 3 — offload engine (the point of the project)
- [x] O1 CPU expert kernel (AVX2, int4 gs64, P-cores), checked vs reference
- [x] O2 Hybrid scheduler: resident experts on GPU, misses on CPU, overlapped per layer
- [ ] O3 Residency policy: LRU baseline → decayed-frequency (τ sweep)
- [ ] O4 Prefetch on copy stream: cross-layer Markov predictor, PCIe budgeted
- [ ] O5 Route traces for this model + port `simulate_policies.py` (old repo) to 256 experts

### Phase 4 — head-to-head
- [ ] B1 colibri CUDA tier on the same container (already on disk)
- [ ] B2 llama.cpp / ik_llama `--n-cpu-moe` on a Qwen3.6-35B-A3B GGUF (needs download)

### Upgrades (only after Phase 2 works)
- [x] U1 Dense trunk requant F16 → int8 (then int4 gs64): frees ~2 GB VRAM for experts, halves bytes/token
- [ ] U2 dp4a / int8-activation expert GEMV
- [ ] U3 CUDA graphs per decode step
- [ ] U4 Prefill path (batched GEMM, chunked DeltaNet)
- [ ] U5 Sampling (temperature / top-p), chat template, server

## Status

| Item | Correct | Bench | Notes |
|---|---|---|---|
| P0.1 ref | greedy: "The capital of France is" → " Paris.\nThe capital of France is Paris." | 0.34 tok/s (oracle only) | not yet cross-checked against colibri logits |
| P0.2 golden | 12 positions, 5678 files, 66 MB | — | stateful ops are tested by chaining positions 0..11 |
| K1 rmsnorm | exact (0 err) | 9 µs | launch-bound; fuse later |
| K2 gemv F16 | rel 2e-7 | qkv 155 GB/s (83%), out_proj 131 (70%), lm_head 163 (87%) | L2 flushed before every timed run |
| K3 router | top-8 ids identical L0/L3/L39 | 21 µs | |
| K4 experts int4 | moe_out rel 1e-7 | 8 experts 0.146 ms, 97 GB/s (51%) | main upgrade target (U2) |
| K5 shared expert | (in moe_out) | 71 µs, 89 GB/s | 6 launches; fuse/graph later |
| K6+K7 attention | rel 5e-7 over 12 chained positions | layer 0.337 ms (86%); core T=4096 45 GB/s | split-K needed for long context |
| K8 DeltaNet | rel 5e-7 over 12 chained positions | layer 0.430 ms (88%); core 32 µs | state column held in registers |
| K9 argmax | 12/12 | 20 µs | |

**Phase 1 roll-up (all experts resident, F16 dense):** 30×0.430 + 10×0.337 + 40×0.255 (router+experts+shared+norms) + 6.25 (lm_head)
≈ **32.8 ms/token → ~30 tok/s** before fusion/graphs. Dense F16 is ~70% of it, which makes U1 (int8/int4 dense) the biggest lever.

**Phase 2 (measured, `python moex.py "..." --chat --n 96`):**

| | |
|---|---|
| E3 check | `moex --check ref/golden`: 12/12 greedy tokens identical, worst logits rel 3.3e-6 |
| Load | 1.7-4.1 s from NVMe; **542 expert slots** fit (0.96 GB, 5.3% of experts) with F16 dense |
| Decode, warm page cache | **11.2 tok/s**: ~30 ms compute + 59 ms blocking expert copies (45% hit, 175 misses/token) |
| Decode, cold page cache | 3.4 tok/s (misses read from SSD) |
| Prompt | token-by-token, 9 tok/s (no prefill path yet) |

**U1 int8 dense (measured):**

| | F16 | int8 |
|---|---|---|
| K10 kernel vs exact int8 math | — | rel 1e-7 |
| lm_head / qkv / out_proj | 6.25 / 0.216 / 0.128 ms | **2.77 / 0.124 / 0.072 ms** (98% / 72% / 62% of 188 GB/s) |
| `--check` vs reference | KL 1e-11 (exact) | 12/12 greedy identical, KL typ. 1e-3, worst 0.014 (pos 0) |
| Expert slots | 542 (5.3%) | **1620 (15.8%)** |
| Expert hit rate (LRU) | 45.2% | **75.5%** |
| Decode, warm | 11.2 tok/s | **19.2 tok/s** = ~20 ms compute + 32 ms blocking misses |

**O1 + O2 hybrid CPU/GPU MoE (measured):**

| | |
|---|---|
| K14 CPU expert (AVX2, fp32) | rel 2e-7 vs GPU; 0.055-0.07 ms/expert from DRAM (8 threads, L3 flushed) |
| Page-fault finding | first touch of a mapped expert 2.3-2.5 ms vs 0.06 ms touched -> experts read + faulted in at load, min working set raised to 19.2 GB |
| Startup | 13 s (18.1 GB experts into RAM; was 25 s before sequential cache fill) |
| Benchmark | `python moex.py "Write a short story about a robot who learns to paint." --chat --n 128` |
| Decode | **30.8-33.3 tok/s**, hit 85% (LRU + background admission), 47 misses/tok on CPU = 7.5-10 ms/tok |
| Correctness | `--check` PASS (int8 KL<=0.014; F16 KL 9e-12); F16 and int8 runs produce the identical 128-token text |
| Known | greedy output loops on this prompt in both F16 and int8: model behavior, needs sampling (U5/K19) |

