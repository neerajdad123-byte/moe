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
- [x] B2 llama.cpp on a Qwen3.6-35B-A3B GGUF (default settings)

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
| K6+K7 attention | rel 5e-7 over 12 chained positions | layer 0.337 ms (86%); core T=4096 45 GB/s | flash-decoding: core T=4096 0.13 ms (16 splits) |
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

**Step 3 + kernel pass (measured):**

| Change | Result |
|---|---|
| Batch benchmark (80 prompts, one warm process) | baseline 35.6 tok/s, 80.5% hit |
| `tools/sim.py` (replays real traces, PCIe budget, online learning) | LRU 80.8% (engine 80.5%, so the sim is faithful); decayed-freq tau=10 tokens **82.0%** (best); tau>=200 loses badly; Markov next-layer prefetch: no gain (35% useful, evicts better experts); Belady ceiling 90.7% |
| Fused projections + tau=10 eviction | dn in-proj 80% / attn qkv 75% of bandwidth (were 72% / 62%) |
| GPU-side hit routing (device slot table) | warm 37.6 -> **41.1 tok/s**, output token-identical |
| CUDA graph per layer | no change (launch overhead was not the limit); kept: frees the host thread |
| nsys timeline (`tools/gaps.py`) | GPU 83% busy; dense GEMV 10.4 ms/token vs 10.2 ms roofline (1.92 GB at 188 GB/s) |
| int4 gs64 dense trunk (tried, **rejected**) | KL 0.02-0.16 typ., 1.2 at pos 0, greedy flip even with int8 lm_head (int8: KL ~1e-3). Kernel also only 22 GB/s before tuning |

**Host/GPU pipeline pass (measured, 80-prompt batch = realistic number):**

| Change | 3x story warm | 80-prompt batch |
|---|---|---|
| start of phase (LRU, separate projections) | 37.6 | 35.6 |
| + fusion, tau=10 eviction, GPU-side hit routing | 41.1 | 34.7 -> (GPU routing not yet in batch run) |
| + zero-copy routing/CPU result (no memcpy ops per layer) | 51.4 | 47.0 |
| + zero-copy embedding/token id | 54.1 | - |
| + expert kernels: padded smem (bank conflicts), persistent grid | 55.0 | **51.4** (81.4% hit) |

Findings: spin-wait scheduling no gain; CPU thread count 8-14 flat; CPU expert kernel compute-bound per core
(6-7 GB/s single thread, ~31 GB/s at 8-12). Per layer the critical path is GPU pre-routing work (dense GEMV at roofline)
+ max(GPU experts+shared, CPU misses), now roughly equal. Ceiling of this design ~57 tok/s warm.
Nsight Compute counters need Windows permission (ERR_NVGPUCTRPERM) - user setting.

## Head-to-head vs llama.cpp (2026-10-12)

Same model (Qwen3.6-35B-A3B), 10 never-seen prompts (`data/h2h_prompts.txt`), same chat template, greedy, 128 tokens,
fresh process per prompt (`tools/h2h.py`). llama.cpp b11563 official CUDA build at default settings, ggml-org Q4_K_M GGUF
(experts Q4_K int4, trunk Q8_0 int8, lm_head Q6_K). moex: colibri int4 gs64 container, int8 trunk.

| | moex v2 | llama.cpp (default) | ratio |
|---|---|---|---|
| decode, fresh process per prompt (mean of 10) | **46.2 tok/s** | 20.3 tok/s | **2.3x** |

Verdict: decode clearly faster (expert-granular residency + CPU misses). Next prefill work: bigger chunks, faster GEMM,
keep the PCIe copy engine saturated.

## Router fix + mixed-precision decision (2026-10-12)

**Bug found by comparing text with llama.cpp:** the top-8 router weights must be renormalized (norm_topk_prob).
colibri's converter wrote false into its meta (the HF config omits the key; the Qwen3.5-MoE default is true). Our engine and
our reference both degenerated into loops; after the fix the output matches llama.cpp's nearly word for word.
All golden data regenerated; new 107-position teacher-forced eval set `ref/eval1`; `--check` is statistical for quantized
weights (int8: mean KL 1.4e-3, top-1 98.6%; F16: exact, KL 7e-13).

Mixed precision on the corrected model, 72 teacher-forced positions + 10-prompt warm decode:

| int4 groups | mean KL | top-1 | slots | decode |
|---|---|---|---|---|
| none (int8) | 1.4e-3 | 98.6% | 1589 | 45.8 tok/s |
| sh_gu | 3.4e-3 | 98.6% | 1609 | 45.5 |
| dn_in | 3.7e-2 | 88.9% | 1771 | 50.4 |
| sh_gu,dn_in,attn_qkv,attn_o | 5.8e-2 | 90.3% | 1853 | 53.1 |

Decision: **int8 stays the default** (int4 DeltaNet input projections flip ~1 token in 9). `--q4` kept as opt-in.
Routing changed with the fix (10-prompt hit 76% -> 71%): traces must be re-collected and the cache policy re-tuned.

## Launch checklist
1. [x] re-collect traces on the fixed model, re-tune eviction/prefetch, re-run head-to-head vs llama.cpp
   sim on corrected traces: tau=10 still best (73.2% vs LRU 72.3%), Markov prefetch no gain, Belady 85.6%.
   10 prompts: moex 41.1 cold / 42.6 warm tok/s vs llama.cpp default 20.3 (2.0x)
2. [x] sampling (temperature, top-p, top-k, repetition penalty, seed); greedy stays the engine default
3. [x] multi-turn chat keeping KV/DeltaNet state (`moex chat`)
4. [x] OpenAI-compatible server (`moex serve`, port 8765, streaming, prefix reuse, stop strings)
5. [x] C++ tokenizer, no Python needed (identical to HF on 54 cases / 66k tokens)
6. [x] other GPUs: fat binary sm_75/80/86/89/90/120 + PTX (4.5 MB exe), slot cap for big GPUs, RAM check (no locking below ~23 GB), AVX2 check
7. [x] `moex download` (WinHTTP, resumable, size-checked) into models\ next to moex.exe; missing model -> clear message
8. [x] long context: int8 KV cache (KL 2.95e-3, top-1 unchanged) + flash-decoding attention (16 key splits, online softmax in
   2048-key tiles, fixed 8 KB smem). Core T=4096: 0.49 -> 0.13 ms/layer. 10-prompt decode: 46.2 fresh process / 47.3 warm (was 41.1 / 42.3).
   26.5k-token prompt: prefill 107 tok/s, decode 29.9 tok/s, coherent (was impossible: score row in smem capped ~24k)
9. [x] README (results, quick start, API, how it works, accuracy, limits), package.bat -> dist/moex-windows-x64.zip (1.2 MB; needs only the NVIDIA driver)
10. [x] `--load` warm start: per-expert use counts kept in moex_hot.bin (model folder; release ships one from 80 prompts),
    hottest fill VRAM at load, RAM locked before the prompt. First reply 44.5 tok/s (was ~6-9 while RAM filled).
    RAM/VRAM free printed before/after load; per-reply tok/s in chat and serve.

## After v0.1.0: studies on route traces (no model run)

**MTP, 1 draft token/step** (`tools/mtp_sim.py build/routes_v2.bin`): per-layer time model calibrated to the measured
46.2 tok/s (dense+overhead 0.368 ms/layer, CPU expert 0.071 ms, GPU expert 0.012 ms), cache replayed with the union of
both tokens' experts (hit 72.8% -> 66-70%, CPU misses 87 -> 154-175 per step), MTP head ~ one more layer, 0.4 ms state
snapshot for rollback.

| draft acceptance | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| tok/s, batched CPU kernel (2nd token +30%) | 54.5 (1.18x) | 56.9 (1.23x) | 59.4 (1.29x) | 62.0 (1.34x) |
| tok/s, no batching (2nd token +100%) | 53.4 (1.16x) | 55.6 (1.20x) | 58.0 (1.26x) | 60.5 (1.31x) |

Verdict: worth building. Dense weights are read once for both tokens, which outweighs the extra misses. Blocker: the
int4 container has no MTP head, so it needs our own int4 conversion from the official BF16 weights.

**RAM tier for 16 GB machines** (`tools/ram_sim.py build/routes_v2.bin`): SSD reads on the critical path per token
(CPU misses whose expert isn't in RAM), scored on prompts 40-79, static ranks from prompts 0-39:

| RAM for experts | 4 GB | 6 GB | 8 GB | 10 GB | 12 GB | 14 GB | 18.2 GB |
|---|---|---|---|---|---|---|---|
| LRU (~ Windows file cache) | 49.3 | 32.8 | 20.8 | 12.7 | 7.3 | 3.6 | 0.1 |
| decayed freq, tau 200 | 43.3 | 30.2 | 19.9 | 12.2 | 6.6 | 3.4 | 0.1 |
| lock a fixed top-R set | 64.9 | 53.4 | 41.3 | 30.0 | 21.2 | 12.8 | 0 |

Verdict: locking a hot set is the worst policy; the file cache's recency is close to the best. What costs is each read:
a page-faulted expert ~2.4 ms (K14) vs ~0.3-0.5 ms as one large NVMe read. Change (untested on hardware): in low-RAM
mode each layer's missing experts (decode, prefill, admissions) are requested together with PrefetchVirtualMemory as
soon as routing is known, and the most-used experts that fit in free RAM are read into the file cache at load (not
locked). `MOEX_LOW_RAM=1` forces the mode on a big machine to exercise the path. Expected at ~8 GB for experts:
~20 reads/token, from ~48 ms/token of faults toward ~8-10 ms: to be measured on a 16 GB machine.
Also fixed: use counts (moex_hot.bin) are now always merged with the saved file; before, a session without --load
overwrote them with its own counts only.

## MTP in the engine (opt-in: MOEX_MTP=1, needs mtp.safetensors from tools/fetch_mtp.py)

Reference (tools/mtp_accept.py, fp32, our int4 MTP experts): draft acceptance 89.6% greedy (pre-final-norm hidden
state; post-norm 88.9%). Engine: 86.5-92% on the 10-prompt set; 9/10 outputs identical to plain greedy (one near-tie).
Speed, back to back on the same (thermally slower) machine state: 42.7 tok/s off, 40.1 on. A round (2 tokens,
1.92 kept on average) costs ~46 ms: 40 layers 36.3 (CPU experts 18.1, GPU routing wait 8.5, ~10 launch/host
overhead), lm_head 2.9, MTP layer 3.9, rejected-draft redo 2.3, state snapshot 0.9.
Why it doesn't pay yet: two adjacent tokens share few experts, so a round misses ~1.9x as many experts as one token
(hit 70.6% -> 66.9%) and the CPU work nearly doubles; the dense-weight saving (gemv_i8 NT=2 reads W once) is smaller.
To reach 70: CUDA graphs for the 2-token layers (~-8 ms), CPU misses at decode efficiency + more VRAM hits for the
union (-8..10 ms), redo without a full step (chunked DeltaNet saving the state after token 0).

MTP round optimizations (full-speed machine state, 10 prompts): off 47.5 tok/s; MTP 43.8 with CUDA graphs for the
pre-routing part of each 2-token layer (40 layers 36.3 -> 32.4 ms); 47.9 with the routing signalled by an event so the
shared expert overlaps the CPU work (layers 28.9 ms, CPU experts 13.5, GPU wait 11.0; round 38.8 ms for 1.86 tokens).
Draft-vocab trimming rejected: 5.4% of generated tokens have ids >= 65536. Plain decode is itself run-to-run
non-deterministic in late tokens (CPU vs GPU rounding depends on which experts are resident): 3/10 prompts split.

## Aggressive VRAM mode: measured, small lever

More expert slots (MOEX_RESERVE_MB shrinks the cache to measure the slope; 10 prompts): 1623 slots 47.9 tok/s, 1473 47.2,
1321 45.9. Simulator agrees: ~0.044 CPU misses/token less per extra slot (cap 1623 -> 1911: 84.3 -> 73.4). So +288 slots
(moving all of lm_head out of VRAM) would give ~+4%, while lm_head on the CPU costs ~8 ms/token. Shared experts or KV in
RAM free 71 / 12-47 slots for extra CPU work every token: net loss. Decode is bound by the dense weights (~1.9 GB/token
= ~10 ms at 188 GB/s of the 21 ms), not the experts.
int4 gs64 per group, alone (ref/eval1, int8 = mean KL 3.0e-3, top-1 98.6%): lm_head KL 8.0e-3 / 95.8%, dn_out 6.1e-3 /
98.6%, attn_qkv 1.3e-2 / 98.6%, attn_o 7.8e-3 / 100%. All above the 5e-3 bar: plain symmetric gs64 is too coarse for
the trunk; a better 4-6 bit scheme is the lever (and MTP, which reads the dense weights once per 2 tokens).

MTP rollback without redo: the DeltaNet kernels (dn_conv_k / dn_recur_k) write the state after the round's first token
into snap as they go; a wrong draft copies it back (no pre-round snapshot, no re-run of the token). Round: snapshot
0.87 -> 0, redo 2.15 -> 0.13 ms. 10 prompts: plain 47.7-47.9 tok/s, MTP 51.1-52.6 (+7-10%), acceptance 86.6%.
Verified: --check with MOEX_MTP=1 runs teacher-forced MTP rounds (true next token as the draft: both rows and the
rollback compared with the reference): mean KL 3.01e-3, top-1 98.6% (plain 2.89e-3, 98.6%). Forced wrong draft
every round (1234 rollbacks): 8/10 outputs identical to plain, the 2 splits at the same near-ties as normal MTP.
Fixed: the draft head wrote into logits[] (the real model's scores) after prefill; it has its own buffer now.

## 16 GB machines: RAM tier study (tools/prefetch_sim.py, tools/ssd_bench.py, tests/ssd_test.cpp)

Cold expert reads on this laptop (C, overlapped, unbuffered): one whole-expert read 0.7-0.9 ms; split into 2-16 pieces
no reliable gain (0.7-2.4 ms, varies with the SSD's power state). Background throughput 3.8-4.2 GB/s (~47 experts per token).
Simulated tok/s with RAM for experts 6 / 8 / 10 GB: recency (today) 16.3 / 20.2 / 25.0, static top-R 12.7 / 14.4 / 16.9,
recency + lag-2/4 co-occurrence prefetch 16.2 / 20.0 / 24.6 (only ~1/3 of prefetches used). Top-N coverage: 90% of
expert uses = 4,891 experts (8.7 GB), 95% = 6,026 (10.7 GB). SSD waits/token ~ 136 e^(-0.24 GB); tok/s ~ 1000 /
(21.6 + t_read x waits). The lever left: the engine reads missing experts through the file map (~2.4 ms, K14); a
private RAM cache with direct reads (~0.8 ms) would be ~+30% at 8 GB. Startup tip printed in low-RAM mode.
