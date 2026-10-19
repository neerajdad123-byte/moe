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

Re-run 2026-10-15 with `tools/h2h.py` (no llama.cpp flags at all): 17.0 tok/s mean (8.2-20.9), so 2.3x is conservative.

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

## Bandwidth and Windows audit (2026-10-15, RTX 4050 Laptop 6 GB, i7-13650HX, DDR5-4800 x2)

Peaks (read-only probes): VRAM 190 GB/s (spec 192); RAM 55-58 GB/s all threads, one P-core 20.4, one E-core 8.5.
Decode, one prompt, 256 tokens, 47.3 tok/s = 21.1 ms/token (nsys, kernels inside graphs):

| per token | ms | note |
|---|---|---|
| dense int8 GEMV (gemv_i8_k, ~1.9 GB) | 10.9 | ~175 GB/s = 91% of measured VRAM peak |
| GPU experts (gate_up + down) | 2.6 | |
| other kernels (DeltaNet, norms, router, attention) | 1.9 | |
| GPU idle, waiting for the CPU misses | 6.2 | 29% of the token |

CPU misses: 86/token x 1.77 MB = 152 MB in 6.0 ms = 25 GB/s, under half the RAM peak (~2 experts per layer split over
8 threads, two phases with a barrier between: latency-bound, not bandwidth-bound). Windows: GPU never throttled
(2730 MHz, memory 8201 MHz, 55 W, 68 C, PCIe 4.0 x8, no clock-event reasons); P-cores ~4.2 GHz under load.
Thread placement (MOEX_CPU_THREADS x process affinity, 2 runs each): all cores 8 threads 47.4, 12 threads 47.8, 6 threads
46.2; P-cores only 8 threads 45.5, 6: 45.8, 10: 45.3, 12: 33.6; one thread per P-core (6): 29.5. The E-cores carry the
loader/copier/main threads, so the default (8 threads, no affinity) is right. `--profile` crashes at moex.cu:532
(cudaEventRecordExternal is only valid inside graph capture; --profile runs the layers uncaptured); nsys works.

## 16 GB machines: router-ahead prefetch + exclusive VRAM/RAM (2026-10-15, simulated on a new trace)

Trace: `moex --batch build/batch_ids.txt --n 128 --trace build/routes_v3.bin --trace-ahead build/ahead_v3.bin`
(80 prompts, 12,864 tokens; per layer l the top-16 of layer l+k's router on layer l's residual, normed with l+k's
post norm). Share of layer l+k's 8 experts in that top-N, decode tokens of prompts 40-79 (k=0 = 100%, self-check):

| k | top-8 | top-12 | top-16 |
|---|---|---|---|
| 1 | 79.7% | 90.6% | 94.3% |
| 2 | 73.0% | 84.3% | 89.2% |
| 3 | 68.8% | 80.1% | 85.5% |
| 4 | 65.4% | 76.6% | 82.2% |

`tools/prefetch_sim.py` (FIFO SSD: direct read 0.8 ms, 0.46 ms/expert in flight, <= 8 prefetches in flight, demand
reads queue behind them; token without SSD waits 21.6 ms), tok/s by RAM for experts:

| policy | 6 GB | 8 GB | 10 GB |
|---|---|---|---|
| today (Windows file cache, demand reads) | 16.0 | 20.0 | 25.0 |
| own RAM cache + direct reads | 24.1 | 28.5 | 33.0 |
| + exclusive VRAM/RAM (admitted leaves RAM, VRAM victims copied back) | 30.5 | 34.9 | 39.3 |
| + router-ahead prefetch k=2 top-8 | **34.8** | **38.7** | **41.9** |
| same, slow SSD (1.2 ms read, 2 GB/s) | 27.0 | 33.2 | 38.8 (today 12.0 / 15.7 / 20.6) |

Without exclusivity: prefetch k=2 top-8 27.6 / 32.9 / 37.2; k=3 slightly worse (fewer late, fewer hits); top-12
worse (56 reads/token at 6 GB, the extra reads evict useful experts). Cycle-aware eviction (evict the layers just
processed first, among the 64 or 512 least recent) changes nothing (+-0.1 tok/s): recency already does it. Exclusivity
is the biggest single lever: today RAM keeps copies of the ~2.9 GB in VRAM (8 GB: SSD waits 21.6 -> 10.3 per token).
At 8 GB the best policy reads 10 prefetches + 5.6 demand reads per token (~28 MB). Not modelled: the VRAM->RAM copy-back
(~58 evictions/token, ~100 MB/token D2H, background), I/O submission cost, the chunked prefill.
No predictor, far ahead (`+far D N`: at layer l, prefetch the N most-used experts of layer (l+D) mod L that are in
neither VRAM nor RAM): 8 GB, on top of own cache + victim (34.9): D=15 N=2 26.4, D=15 N=4 20.9, D=5 N=4 21.1, D=30 N=4
20.6 tok/s. The popular experts are already cached, so it reads 70-90 rarely-used experts per token (~1 in 3 used) and
the demand reads queue behind them. Restricted to each layer's top 8-16 it reads nothing (all resident): no change.
Adding it to the router-ahead prefetch: 38.7 -> 27.0. Streaming every expert of each layer: 18.1 GB per token = 0.21 tok/s.

### Real runs (2026-10-15): own RAM cache (MOEX_RAM_GB), victim copy-back (MOEX_VICTIM), router-ahead (MOEX_AHEAD=2)

This laptop with `build\ram_balloon.exe` locking RAM so X + 2 GB is free at start (a 16 GB machine's situation),
MOEX_LOW_RAM=1, `--batch build/ram3_ids.txt --n 128 --load` (3 everyday prompts, 381 decode tokens), tok/s:

| X = RAM for experts | 6 GB | 8 GB | 10 GB | sim (6/8/10) |
|---|---|---|---|---|
| today (Windows file cache) | 24.3 | 27.9 | 29.3 | 16.0 / 20.0 / 25.0 |
| own cache, direct reads | 16.9 | 24.4 | 30.9 | 24.1 / 28.5 / 33.0 |
| + victim (VRAM/RAM exclusive) | 26.5 | 30.8 | 34.8 | 30.5 / 34.9 / 39.3 |
| + ahead k=2, no victim | 21.1 | 26.8 | 33.0 | 27.6 / 32.9 / 37.2 |
| + victim + ahead k=2 | **27.9** | **32.5** | **36.2** | 34.8 / 38.7 / 41.9 |

Output identical to the full-RAM run (8 GB, all on: 128/128, 123/128, 128/128 tokens; the split is a known near-tie).
Per token at 8 GB, all on: 4.1 demand reads, 8.4 prefetched (4.9 used), 70.5 copied back, 5.5 ms waiting for reads.
Why below the simulation: (1) a demand read costs ~1.1-1.3 ms here, not 0.8 (two I/Os per expert, Q and S apart in
the file); (2) the file cache gets more than X (Windows lets it grow into the 2 GB margin) and is faster than the
simulated 1.68 ms per read, so "today" is the stronger baseline; (3) overheads outside the read wait: token time minus
wait is 21.2 ms with the own cache alone (= full-RAM speed), 23.1 with victim (~2.3 ms: 3 CUDA calls per copy-back on
the main thread, ~70/token), 25.1 with ahead (~4 ms: the predictor's 4 kernels run before the routing event, on the
critical path). Fixes: enqueue copy-backs on the loader thread; run the predictor after ev_route (the GPU idles
while the CPU does the misses) and read it a layer later with k=3.

Both fixes (copy-backs enqueued by the loader thread; predictor after the layer's GPU work, k=3, read a layer later),
same matrix, new session (today's baseline moved by -1.8..+1.3 between sessions):

| X | 6 GB | 8 GB | 10 GB |
|---|---|---|---|
| today | 23.8 | 26.1 | 30.6 |
| own cache | 19.3 | 25.0 | 30.8 |
| + victim | 25.4 | 30.3 | 32.3 |
| + ahead k=3, no victim | 20.2 | 25.6 | 32.3 |
| + victim + ahead k=3 | **26.4** | **30.7** | **34.6** |
| + victim + ahead k=2 (read a layer later) | 25.8 | 30.6 | 32.9 |

No net gain over the first version (27.9 / 32.5 / 36.2): the time outside the read wait fell (8 GB, all on, no balloon:
25.3 -> 22.1 ms/token) but the wait grew by as much. Wait breakdown (8 GB, all on, no balloon, per token): demand reads
4.98 ms (4.8 x ~1.0 ms: a real direct read costs ~0.9-1.0 ms, near the simulated 0.8), copy-backs not done yet 2.18
(the CPU needs an expert evicted from VRAM whose copy-back still queues behind the loader's admissions), late
prefetches 0.34; issuing ReadFile 1.35 ms (~0.09 ms per read, not the bottleneck). Next: copy-backs on their own
stream, enqueued at eviction by a small thread, with the admission's H2D waiting on that event.

## v0.1.3: copy-backs on their own stream; 16 GB mode and MTP on by default (2026-10-17)

Copy-backs (VRAM victim -> own RAM cache) now go out on their own stream (cb) from a small thread at eviction, with
no wait (the slot holds a finished expert); the admission's H2D waits on that copy (back_done) before overwriting the
slot. Measured with the matrix below: the copy-back wait didn't shrink (6/8/10 GB: 2.33 / 1.91 / 2.94 ms/token vs 2.18
before). The bottleneck is the queue, not who enqueues it: ~56-65 copy-backs per token (~100-115 MB D2H) run in order
on one stream, so an expert the CPU needs right after its eviction waits behind the earlier ones. Next: copy back only
experts likely to be used again (score above a threshold), or let the CPU read a still-intact victim slot directly.

16 GB mode on by default: on a low-RAM machine the own RAM cache takes the RAM free at load minus 2 GB (MOEX_RAM_GB=N
overrides, 0 = Windows' file cache as before), exclusive VRAM/RAM (MOEX_VICTIM=0 off), router-ahead prefetch k=3
(MOEX_AHEAD=0 off). Same balloon setup as above (X + 2 GB free at start, MOEX_LOW_RAM=1, ram3_ids, --n 128 --load):

| X = RAM for experts | 6 GB | 8 GB | 10 GB |
|---|---|---|---|
| before (MOEX_RAM_GB=0: file cache) | 21.0 | 26.0 | 28.4 |
| v0.1.3 default (cache size picked: 6.1 / 8.5 / 10.4 GB) | **25.2 (+20%)** | **30.8 (+19%)** | **34.2 (+20%)** |

Output: prompts 1-2 match the full-RAM run (128/128, 123/128: the known near-tie); prompt 3 splits at token 5, a
near-tie that also splits between two runs of the old mode. A shutdown hang (the backer thread quitting while the
loader waited on one of its copies) was found and fixed: the backer drains its queue before it exits.

MTP on by default for chat / serve / bench when every expert fits in RAM and the head is downloaded (`--no-mtp` off,
`--mtp` forces it). Low-RAM machines keep it off: the draft token's extra experts become SSD reads, and the own cache
needs it off.
`moex bench` (10 questions, temp 0.7, same laptop, alternating): MTP on 50.2 / 49.3, off 48.1 / 46.9 tok/s: +4-5%
(sampling accepts fewer drafts than the greedy +7-10%). Docs and help now say ~5%.

## Idle CPU at the prompt (issue #1, 2026-10-18)

`moex chat` at the `>` prompt and `moex serve` between requests kept ~7 cores busy (35% of 20 logical CPUs): the decode
CpuPool's workers pause-spun, then `yield()`ed forever. Now a worker pause-spins ~1 ms, yields while a job came within
the last 50 ms (HOT_MS, well above the gap between decode tokens), then blocks on `gen.wait` (WaitOnAddress); the
dispatcher calls `notify_all` only when a worker sleeps, so hot decode adds no wake-up calls. A spin=false pool (the
copy pool outside prefill) blocks right after the pause loop instead of polling every 200 us.
Same laptop: idle at the prompt (before and after a reply) and in serve after a request: 6.9 cores -> 0.00.
`moex bench` alternating before/after: 51.3 / 49.5 vs 50.7 / 50.2 tok/s (noise), first token unchanged; `--check
ref/eval1` PASS. The loader/backer threads still poll with 20-50 us sleeps; that measured as 0.00 cores, so left as is.

## Prefill: profile and plan (2026-10-18)

Goal: prompt reading from ~130-150 tok/s to 1,000+ on the RTX 4050 Laptop 6 GB (Strata reads 1,600-2,600 on a 12 GB
RTX 5070; long prompts are where users feel moex is slow: 16K tokens take 2 minutes today).

### Baseline (v0.1.3, `--batch build/pf_N.txt --n 4 --ctx 17000 --load --no-mtp`; prompts = this repo's text, chat template)

| prompt | tok/s | gpu-wait | streaming-wait | cpu-experts | streamed experts | host memcpy |
|---|---|---|---|---|---|---|
| 4,096 | **147** | 11.4 s | 11.3 s | 2.6 s | 76,810 (120 per chunk-layer) | 9.1 s |
| 16,384 | **128** | 59.9 s | 49.2 s | 9.6 s | 326,464 (128 per chunk-layer) | 36.5 s |

(1,024 tokens: 57 tok/s on a cold start, 1.07 M page faults while --load was still filling RAM: not representative.)

Nsight Systems, 4K prompt, GPU kernel time per chunk-layer (256 rows; the trace covered the first ~128 chunk-layers):

| kernel | ms per chunk-layer | share | note |
|---|---|---|---|
| pairs_gate_up_k + pairs_down_k (experts) | ~24 | 74% | ~2.7 TFLOPS: per-pair GEMV-style, no tensor cores |
| gemm_wmma_k int8 (dense + shared expert) | ~5.6 | 17% | ~3.6 TFLOPS on wmma |
| gemv_f16_k (router, one GEMV per row) | 0.94 | 3% | 256 launches per layer |
| export_route_k | 0.30 | 1% | a copy, should be ~0.02 |
| attention / DeltaNet / norms / reduce | ~1 | 4% | at 4K; attention grows with context (gpu-wait 2.8 -> 3.7 ms/token from 4K to 16K) |

=> ~32 ms GPU per chunk-layer = 1.3 s per 256-token chunk = **~200 tok/s GPU ceiling** before any streaming. PCIe:
each 256-token chunk uses ~all 256 experts of a layer, ~120 of them not resident, so every chunk streams ~8.5 GB
(~33 MB per prompt token); at ~12 GB/s (PCIe 4.0 x8) that alone caps prefill at ~360 tok/s, and the extra host
memcpy (file map -> pinned ring, 4 threads, ~15 GB/s) sits in front of the DMA.

### Plan (each step: `--check ref/eval1` with prefill still PASS, test_kernels vs the current kernel, then 4K/16K tok/s)

- [x] PF1 **Grouped int4 expert GEMM on tensor cores** for prefill: the pairs are already grouped by expert (~8 rows
  per expert at CH=256, ~32 at 1024); per expert, dequant int4 gs64 -> fp16 tiles in shared memory and run mma
  (m16n8k16) over its rows, gate+up+SiLU fused, then down with the routing weight. Target: experts 24 -> <=4 ms per
  chunk-layer. Biggest single lever at every prompt length.
- [x] PF2 **Bigger chunks**: CH 256 -> 1024 (2048 where VRAM allows). Streamed bytes per prompt token fall ~2.5-4x
  (an expert is streamed once per chunk, not once per 256 rows) and every GEMM gets fatter. Scratch is ~290 KB per row
  (~290 MB at 1024 = ~165 expert slots, borrowed from the cache for the length of the prefill as today); alias the
  buffers that are never live together (qkvc/bigc/convc/oc) to cut it. PCIe ceiling at 1024: ~870 tok/s; at 2048: ~1,700.
- [ ] PF3 **Streaming without the extra copy**: measure after PF2 (it shrinks the problem). Options: a bigger pinned
  ring with more copy threads, or reading experts from the own RAM cache straight into pinned memory. (Page-locking the
  file map was ruled out: WDDM budget, see above.)
- [ ] PF4 **Dense GEMM**: gemm_wmma_k at ~3.6 TFLOPS -> int8 activations + IMMA (or better fp16 tiling); target 2-3x.
- [ ] PF5 **Small kernels**: router as one GEMM over the chunk (not T GEMVs), export_route_k as a plain copy,
  pairs_reduce fused into the expert down kernel.
- [ ] PF6 **Long-context attention in prefill**: profile at 16K; the chunk attends token by token with one split.
  A flash-attention style kernel over the chunk (tensor cores, causal within the chunk) if it shows up after PF1-PF2.
- [ ] PF7 **Ship it**: a `moex bench --prefill` line (4K and 16K tok/s), README numbers, release.

Expected after PF1 + PF2 (estimate, to be replaced by measurements): ~500-800 tok/s on 4K-16K prompts; PF3-PF6 toward
1,000+. Order: PF1 first (largest, independent of chunk size), then PF2, then re-profile before PF3-PF6.

### PF1 + PF2 results (2026-10-18)

PF1: `tc_gate_up_k` / `tc_down_k` (src/kernels.cuh): per block 128 weight rows x up to 32 of the expert's rows, each
K step one scale group dequantized to fp16 in shared memory, wmma 16x16x16, the next step's loads in flight during the
MMAs, 16-row tiles when an expert has <= 16 rows, Cs over As (23 KB: 4 blocks per SM). Prefill only; the 2-row MTP
round keeps the GEMV kernels. `MOEX_PF_GEMV=1` switches prefill back for A/B tests.
test_kernels PF1 (layer 0, 256 rows, top-8): vs pairs_forward rel 4e-4 (fp16 operands);
256 experts (~8 rows each, a real chunk): 21.1 -> **4.76 ms** (4.4x); 32 experts (~64 rows each): 20.8 -> **1.14 ms** (18x).
A first version without the register prefetch and with 39 KB of shared memory: 7.46 / 3.09 ms.

PF2: CH 256 -> 1024. Expert slots unchanged (1515: the scratch is borrowed from the cache only during prefill).

| prompt | v0.1.3 | + PF1 (CH 256) | + PF2 (CH 1024) |
|---|---|---|---|
| 4,096 tok/s | 147 | 170 | **300** |
| 16,384 tok/s | 128 | 143 | **228** |
| 4K streamed experts | 76,810 | 76,832 | 27,124 |

PF1 alone gave only +16%: the GPU expert time fell 4x but the streaming (host memcpy into the pinned ring + DMA) was
already as long, so the wait just moved there. PF2 cut the streamed bytes 2.8x.
Checks: `--check ref/eval1` PASS (mean KL 1.48e-3, was 2.89e-3; top-1 98.6%, the same one mismatch); a 4,096-token
prompt (4 chunks) then 48 greedy tokens: identical to `--no-prefill` (token by token), 48/48; test_kernels ALL PASS;
`moex bench` decode 50.3 tok/s (49.0 at the v0.1.3 release: unchanged).
Found on the way: `--check --f16` FAILs even with `--no-prefill` and the old kernels (worst KL 1e-2 vs the 1e-5
limit): the int8 KV cache is on in F16 mode too, so the README's "matches to 1e-12" is stale. Not touched here.

Next, from the 4K run at CH 1024: wall 13.7 s but gpu-wait 0.65 + streaming-wait 3.65 + cpu-experts 0.81 = 5.1 s; the
other ~8.5 s is host time not yet attributed (per-expert lists, `prefetch_experts`, launches, the MTP-off tail?).
Profile that first (it is now the largest part), then PF3 (streaming), then try CH 2048.

## moex serve for agents (2026-10-19)

Tool calling (OpenAI tools / tool_calls, Anthropic tool_use / tool_result), Anthropic /v1/messages + count_tokens,
legacy /v1/completions, stream_options.include_usage, max_tokens default = rest of the context. Prompts are built as
Qwen3.6's chat_template.jinja (tools/check_template.py: identical tokens on 6 cases). tools/test_tools.py with the
official SDKs: 17/17. Fixed on the way: `serve --think` was ignored and thinking came back inside content (issue #2):
now reasoning_content / thinking blocks.

Open: agents send side requests (titles, small checks) with other prompts between main turns. A side generation
replaces the engine state, so the next main turn re-prefills its whole prompt (20K tokens ~ 70 s at ~300 tok/s).
Fix idea: snapshot the state (DeltaNet states ~63 MB + the KV rows) at the end of the longest recent prompt into
pinned RAM, restore it when a request extends that prefix.

## Speed pass (branch `speed`, 2026-10-19): full time breakdown first

Instrumentation: NVTX ranges (`decode`, `mtp round`, `prefill`) + `tools/nsys_breakdown.py` (GPU time per kernel and
grid per range kind, GPU busy vs host wall); `--batch` now prints decode layers split by "all hits in VRAM" vs "with a
CPU miss", MTP layers the same, and prefill's chunk_layer host timeline (enqueue / gpu-wait / group / prefetch /
launch / cpu-experts / cpu-sum / copier-wait / tail).

Baseline (f28d388 + instrumentation), 10 short prompts x 128 greedy tokens (`--batch build/dec10.txt --n 128 --load`):

| | no MTP | MTP |
|---|---|---|
| tok/s | 46.8 (21.4 ms/tok) | 53.7 (37 ms/round, 1.92 tok) |
| expert hit | 67.9% | 61.5% |
| layers with >= 1 CPU miss | 89.1% | 97.5% |
| host us/layer, miss layer | launch 21, wait-routing 253, CPU 178, admissions 25 | gpu-wait 284, CPU 347, rest ~80 |
| host us/layer, all-hit layer | launch 20, wait-routing 274 | gpu-wait 372, rest ~50 |
| GPU busy (nsys) | 16.0 ms/tok of 23.1 wall (69%) | 25.8 ms/round of 39.7 (65%) |

Decode GPU per token (nsys, no MTP): gemv_i8 dn_in 4.32 (30x, ~175 GB/s), lm_head 2.89, out+o+shared-down 2.48 (80x),
qkv 1.08, shared gate_up 0.53 => dense int8 11.3 ms at the roofline; experts gate_up 1.56 + down 0.88; dn_recur 0.52;
rmsnorm 0.32, add 0.31, router gemv 0.28 + topk 0.18 + export 0.27; attention 0.08 (short ctx); rest < 0.1 each.
MTP round: lm_head x2 5.74, pairs_gate_up 4.61 + pairs_down 3.57 (GEMV over 2 rows), dense as above.

Prefill 4K (299 tok/s, 13.71 s): chunk_layer host time 13.55 s = **enqueue 6.77** (host blocked in launches: the
queue fills with 1024 router GEMVs + 1024 shared-gate GEMVs per layer, so it waits for the GPU's pre-routing work),
copier-wait 3.84, launch 1.57, cpu-experts 0.69, gpu-wait 0.54. 16K (224 tok/s, 73.2 s): enqueue **44.6**, copier-wait 16.2,
launch 6.0, cpu 2.7, gpu-wait 2.5. GPU per 1024-row chunk-layer at 4K (nsys, ~38 ms, busy 33% under tracing):
gemm_wmma dn_in 16.8 ms per call (~3.1 TFLOPS), out/o/sh_d 3.0 each, qkv 8.3, sh_gu 1.5; **attn_decode_k 24 ms per
attention layer** (row by row, grows with context); router + shared-gate GEMVs 4.3 ms in 2048 launches; export_route
0.9, pairs_reduce 0.7.

### S1a: router + shared-expert gate as one launch per chunk (gemv_f16 over `rows` via blockIdx.y)
Same per-row math (bit-identical). 4K: 299 -> **373 tok/s** (enqueue 6.77 -> 0.01 s; the wait is now honest GPU time,
gpu-wait 6.13 s, plus copier-wait 3.80). 16K: 224 -> **283** (gpu-wait 40.8 s: mostly row-by-row attention).
`--check ref/eval1` PASS (mean KL 1.48e-3, top-1 98.6%, unchanged); test_kernels ALL PASS.

### S1b: stream ahead (one streamer thread per prefill, VRAM ring of 8 x 24 experts)
At >= 256 rows per chunk ~every expert of a layer is used (~32 rows each), so the chunk's start queues every layer's
non-resident experts and the copies run under the GPU's earlier work (no wait for the routing; the per-layer copier
thread is gone; experts with 1 row are now GPU work too). Smaller chunks queue each layer's list after its routing as
before. The ring borrows ~250 MB more cache during prefill; GPUs whose cache would be < 3x the borrow keep 2 batches.
4K: 373 -> **525 tok/s** (copier-wait 3.80 -> 0.72 s; streamer busy 6.3 of 7.8 s: memcpy 3.5 + DMA wait 2.8; streamed
26.7K -> 36.4K experts: the bigger ring leaves fewer resident). 16K: 283 -> **344** (gpu-wait 43.4 s = attention + GEMMs).
Small cache (520 slots, 2 batches): 335 tok/s. Same greedy tokens; `--check` PASS (mean KL 1.49e-3, top-1 98.6%).

### S1c: int8-weight prefill GEMM, 128x128x32 tiles (gemm_i8_k)
8 warps of 32x64, 16-byte loads, next K step's loads in flight. test_kernels K12 (now also S=1024): qkv 8192x2048
S=1024 **1.83 ms = 18.8 TFLOPS** (old gemm_wmma_k: ~3.7 TFLOPS; S=256 2.35 -> 0.70 ms). Prefill 4K: 525 -> **644 tok/s**
(gpu-wait 6.72 -> 3.73 s; now the streamer is the limit: memcpy 3.73 + DMA wait 1.80 of 6.36 s). 16K: 344 -> **362**
(gpu-wait 32.8 s: attention). Same greedy tokens; `--check` PASS (mean KL 1.48e-3, top-1 98.6%); test_kernels ALL PASS.

### S1d: prefill chunk attention on tensor cores, GQA (attn_prefill_k)
One block per (kv head, 4 tokens) = 32 query rows (8 heads x 4 tokens) sharing each 32-key K/V tile; S = Q.K^T and
O += P.V on wmma (int8 K/V exact in fp16, P carries V's scale), online softmax fp32, O in shared memory (90 KB).
Chunks of > 2 rows only (decode and the MTP round keep attn_decode_k). test_kernels K7P vs per-row kernel: rel <= 1.9e-3;
1024 rows at pos 3072: 57.1 -> 9.4 ms; at pos 15360: 259 -> 39.6 ms (~6.7 TFLOPS, room left).
A/B alternating (old, new, old, new; laptop warm, so lower than single runs): 4K 593 / 584 / 551 / 575 (no change: little
attention at 4K); **16K 367 / 576 / 354 / 575 (+60%)**. Single runs: 4K 712, 16K 600. Same greedy tokens in all runs.
`--check` PASS: mean KL 1.48e-3 -> **1.71e-3** (fp16 Q and P), worst 1.79e-2, top-1 98.6%. test_kernels ALL PASS.
`moex bench` decode: 49.1 (old) / 48.7 (new) avg, unchanged.

### S2: the host copy (user's step 2): measured PCIe-bound; non-temporal stores
nsys, 4K: the H2D engine is busy 6.91 s of the 7.01 s prefill (64.9 GB, 9.4 GB/s while busy): prefill is PCIe-bound,
kernels busy 3.72 s. Ceiling test (throwaway: skip the memcpy, DMA stale data): 580 -> **727 tok/s** (DMA 11.8 GB/s):
the copy slows the DMA by contending for RAM bandwidth (RAM peak 55-58 GB/s; memcpy 17.5 GB/s = ~52 GB/s with
read-for-ownership). A cudaStreamQuery after each enqueue (WDDM batching): no change (583/583, 582/585). Direct
SSD reads: no (~4 GB/s). Kept: `copy_nt` (AVX2 streaming stores, no RFO): memcpy 3.6 -> 2.9 s; A/B 4K 582/581 ->
595/605 (+3%), 16K 595/574 -> 590/589 (flat). Same tokens; `--check` PASS (mean KL 1.72e-3); test_kernels ALL PASS;
bench 49.7 / 49.0. 800+ needs fewer bytes over PCIe, not a cheaper copy (64.9 GB at 11.8 GB/s caps 4K at ~745).

### S3: where the streamed bytes go (prefill line: GB, H2D busy from timing events, experts by rows used)
4K: 36,360 streamed = 64.3 GB, H2D busy 6.70 s (9.6 GB/s). Rows each streamed expert ended up with: **0: 6,727 (18.5%)**,
1: 2,402 (6.6%), 2: 1,774, 3: 1,417, 4: 1,169, 5+: 22,871. 16K: 145,440 = 257.4 GB, H2D 26.9 s; 0: 23,359, 1: 8,534,
2: 6,173, 3: 4,938, 4: 4,213, 5+: 98,223. Routing is skewed: a quarter of stream-ahead's bytes (0 or 1 row) is waste.

### S4: predicted stream-ahead (layer l+1's router on layer l's residual)
Chunks of >= 256 rows: layer l's pre-routing work also runs layer l+1's router on its residual (normed with l+1's post
norm; one fp16 tensor-core GEMM + top-8), and right after l's routing the host queues the non-resident experts predicted
for >= 2 rows; real routing decides (unpredicted >= 2 rows: on demand; 1 row: CPU). Layer 0 and the MTP layer: on demand.
4K: 64.3 -> **50.9 GB**, H2D busy 6.70 -> 5.36 s; streamed experts with 0 rows 6,727 -> 662, 1 row 2,402 -> 920.
16K: 257.4 -> 211.7 GB, H2D 26.9 -> 22.2 s. A/B alternating: **4K 612 / 727 / 598 / 723**, **16K 589 / 679 / 594 / 681**
(+20% / +15%). Same greedy tokens; `--check` PASS (mean KL 1.71e-3, top-1 98.6%); test_kernels ALL PASS; bench 49.3 / 48.2,
rerun in reverse order 48.4 (new) / 48.3 (old): unchanged (the second run of a pair is the warmer one).

### S5: CPU threshold for prefill experts (STREAM_MIN 2 -> 3: experts with <= 2 rows run on the CPU)
Sweep, single runs at 4K: CPU <= 1 row 503 (taken while the machine was disturbed: CPU experts 2.68 s vs 0.45 normal),
<= 2 rows 803 (48.0 GB, H2D 4.76 s), <= 3 rows 798, <= 4 rows 821; stopped there (16K points not run). Kept <= 2.
A/B alternating (old, new, old, new): 4K 748 / 765 / 728 / 761, 16K 681 / 709 / 682 / 710 (+3-4%). Same greedy tokens;
`--check` PASS (mean KL 1.26e-3, worst 2.99e-2: more singletons on the CPU's fp32 path, top-1 98.6%); test_kernels ALL
PASS. Power: on AC; Windows reported the Balanced plan with no performance overlay during these runs.

### Parked: CH 2048 (not built)
Each expert would cross PCIe once per 2048 rows instead of 1024 (the remaining limit: H2D busy ~= the prefill wall).
Design: alias the scratch that is never live at once in stream order: the mixer buffers (attention: qkvc, qoc, agc;
DeltaNet: bigc, convc, oc) and the MoE pair buffers (actp, ypair) share one region of max(17408, 24640, 20480) floats
per row; per-row scratch ~298 KB -> ~147 KB, so 2048 rows borrow what 1024 rows borrow today. Make CH a runtime value
(2048; 1024 when the cache would be < 3x the borrow, like the ring), mtp_cat allocated for 2048. Check pairs_forward_tc
with ~64 rows per expert and the K12 GEMM at S=2048.

### Prefill summary (`--batch build/pf_N.txt --n 4 --ctx 17000 --load --no-mtp`)
Single runs unless marked A/B (A/B = the new side of alternating runs; the laptop runs slower when warm).

| commit | change | 4K tok/s | 16K tok/s | mean KL |
|---|---|---|---|---|
| f28d388 | baseline | 299 | 224 | 1.48e-3 |
| 5e12c13 | profiling (NVTX, per-kernel / per-phase breakdown) | - | - | - |
| 5fab266 | router + shared gate: one launch per chunk | 373 | 283 | 1.48e-3 |
| a4ecff6 | one streamer thread, stream ahead | 525 | 344 | 1.49e-3 |
| 8dd7290 | int8 GEMM 128x128 tiles (18.8 TFLOPS) | 644 | 362 | 1.48e-3 |
| 71ca6f9 | tensor-core GQA prefill attention | 712 (A/B ~580, flat) | 600 (A/B 575, +60%) | 1.71e-3 |
| 5bc3f70 | non-temporal copy into the pinned ring | A/B 595-605 (+3%) | A/B 589-590 | 1.72e-3 |
| 3be0336 | counters: GB streamed, H2D busy, rows per expert | - | - | - |
| 83103d8 | predicted stream-ahead (layer l+1's router) | A/B 723-727 (+20%) | A/B 679-681 (+15%) | 1.71e-3 |
| (this) | CPU takes experts with <= 2 rows | A/B 761-765 (+4%) | A/B 709-710 (+4%) | 1.26e-3 |

## Decode (speed pass, 2026-10-19)

### D0: breakdown re-measured (d6b63e9, MTP off, nsys)
46.4 tok/s = 21.5 ms/token. GPU kernels 16.1 ms of 23.4 traced: dense int8 GEMVs 11.25 (dn_in 4.30, lm_head 2.87,
out/o/shared-down 2.47, qkv 1.07, shared gate_up 0.53), GPU experts 2.44, other ~2.4; GPU idle on CPU misses ~6.
87% of layers have a CPU miss (host: wait-routing 256 us + CPU 181 us + admissions 26 us per such layer).

### D1: GQA decode attention (attn_gqa_k)
One block per (kv head, token, split) for all 8 query heads: thread i scores key i for the 8 heads (K read once),
warp h runs head h's online softmax, thread d does dim d of P.V for the 8 heads (V read once); attn_combine_k merges.
Splits 16 -> 32 per kv head (ASPLIT). The MTP round's 2-token attention now splits too (it ran 32 unsplit blocks).
test_kernels K7G vs attn_decode_k: rel ~1e-6; per layer: 1K 0.047 -> 0.037 ms, 16K 0.468 -> 0.192, 26K 0.738 -> 0.295.
Decode after a long prompt (`--batch` 1K / 16K / 26K prompts, 128 greedy tokens; 26K = the 16K text repeated):

| tok/s | 1K | 16K | 26K |
|---|---|---|---|
| MTP off, before | 45.2 | 40.3 | 37.4 |
| MTP off, after | 45.8 | 43.9 | 42.7 |
| MTP on, before | 49.9 | 33.8 | 27.2 |
| MTP on, after | 50.2 | **49.1** | **45.4** |

Same greedy tokens (one old-binary MTP run split in late tokens: the known run-to-run near-tie). `--check` PASS (mean
KL 1.26e-3, top-1 98.6%); test_kernels ALL PASS. `moex bench` (short prompts, unchanged): MTP on 49.3 / 49.1 / 50.2 /
49.4, off 47.4 / 47.4 / 47.6 / 47.7 (old, new, old, new).

### D1 verification (same session): per-prompt A/B, acceptance, nsys
`build/abd.sh`: 1 warm-up, then old (d6b63e9) / new (3b4117c) alternating, 2 rounds, 1K / 16K / 26K prompts, 128 tokens.

| tok/s | 1K | 16K | 26K | MTP acceptance 1K / 16K / 26K |
|---|---|---|---|---|
| MTP off, old | 44.5 / 44.9 | 39.3 / 39.2 | 36.6 / 36.6 | - |
| MTP off, new | 45.4 / 45.4 | 42.8 / 43.2 | 42.4 / 41.0 | - |
| MTP on, old | 49.0 / 49.5 | 33.8 / 33.7 | 26.9 / 26.3 | 86.8 / 89.6 / 86.8-81.4% |
| MTP on, new | 50.3 / 49.8 | 46.7 / 49.2 | 44.9 / 45.9 | 86.8 / 89.6 / 84.1-86.8% |

Attention GPU ms (nsys, `tools/nsys_breakdown.py` now groups decode / MTP ranges by prompt): per decode token, MTP off,
old 0.30 / 3.92 / 6.18 -> new 0.18 / 1.58 / 2.49; per MTP round, old 1.36 / **20.3 / 32.3** (the 2-token chunk path ran
32 unsplit blocks over the whole context) -> new 0.29 / 2.34 / 3.62. MTP round wall at 26K: 72.4 -> 45.0 ms.
Greedy tokens: 1K identical in every run; 16K / 26K hashes vary run to run in the old binary alone too (16K: 50edb1,
b043d3, b480a0 across its runs), the known near-tie; the new binary produces the same hashes as old runs.

### D2: re-profile after D1 (MTP off; nsys wall includes ~1.5 ms/token of tracing)
| ms/token | 1K | 16K |
|---|---|---|
| dense int8 GEMV | 11.26 | 11.09 |
| GPU idle waiting on CPU experts (traced) | 7.30 | 7.57 |
| GPU experts | 2.39 | 2.40 |
| attention | 0.18 | 1.58 |
| DeltaNet | 0.58 | 0.58 |
| norms / router / other | 1.66 | 1.64 |
| untraced total | 21.9 (45.6 tok/s) | 23.2 (43.1 tok/s) |
CPU misses 108.6 / 104.0 per token, CPU experts 6.6 ms/token; 90% of layers have a miss (wait-routing 274 us + CPU
184 us + admissions 26 us). Admissions 83 experts/token = ~147 MB H2D per token (~6.6 GB/s at 45 tok/s): churn, see D5.

### D3: CPU experts with int8 activations + AVX-VNNI (decode and the MTP round)
x quantized once per layer (int8 per 64-group, deinterleaved like the nibbles; nibble ^ 8 = w + 8 unsigned, the
8 * sum(x) correction precomputed); vpdpbusd (AVX-VNNI, picked by cpuid; AVX2 maddubs + madd fallback). Two phases:
(expert, 64-row block) items do gate/up and quantize their 64 activations as one group of the down input; then
64-row blocks of down read whole 256-byte rows for every expert. The MTP round's CPU misses (`run_pairs_small`) use
the same kernel over (expert, token) pairs; prefill keeps fp32 `run_pairs`.
First try, fused (each item one column group of down for all rows): CPU 6.2 -> 6.8 ms/token. Phase profile:
down took as long as gate+up (22.1 vs 22.7 CPU-ms/token) at half the bytes: two items read halves of each 64-byte
line of down, so its RAM traffic doubled. Two-phase: down 11.3 CPU-ms (half of gate+up), quantize x 0.42 -> 0.09 ms.
test_kernels K14 (8 threads, from RAM): 1 expert 0.071 -> 0.052 ms, 8 experts 0.345 -> 0.330 (~43 GB/s; RAM ~55).
A/B (old, new, 2 rounds), 10 prompts x 64 tokens: MTP off 47.2 / 47.2 -> 49.3 / 49.3 tok/s, CPU 6.0 -> 5.0 ms/token;
MTP on 55.2 / 54.7 -> 59.3 / 58.6, CPU 6.9 -> 5.65. `moex bench`: MTP on 50.3 / 49.6 -> 53.5 / 53.4; off 48.0 / 47.7 ->
49.9 / 50.1. `--check` PASS: mean KL 1.26e-3 -> 1.52e-3, worst 2.99e-2, top-1 98.6%. Greedy: 8/10 prompts identical
over 64 tokens; 2 split late (tokens 58, 61) at top-2 gaps 0.14 / 0.33 logit (median gap 11.5). Accepted on KL and
top-1 (approximate kernels are gated on those); this build is the new greedy baseline.

### VRAM audit (Checkpoint 1; `[vram]` line at load), ctx 8192, 6140 MB
In use before load 1060 MB: a bare CUDA context already shows it (nvidia-smi: 0 MB): the WDDM driver's reservation,
not ours. lm_head int8 486 | dense int8 + MTP head 1448 / 1496 (routers f16 40) | KV cache 81 (sized for ctx) |
DeltaNet state 63 | decode / MTP scratch 4 / 88 (MTP snapshot 63) | expert cache 1586 slots (MTP off) / 1508 (MTP on)
= 2.7 / 2.5 GB, of which prefill borrows 365 | reserve 320. Embeddings are already in RAM (mapped); prefill scratch is
already borrowed from the cache. Hit rate 68.8% / 61.4%, misses 99.8 / 130 per token (10 short prompts).
Cache-size slope, same binary, MOEX_RESERVE_MB 320 vs 150 (+101 slots), alternating: hit 68.4 / 68.9 -> 69.7 / 70.1%,
misses 101 / 100 -> 96.8 / 95.7, 49.0 / 49.1 -> 49.7 / 49.6 tok/s (+1.2%, below the 3% noise bar).

Budget at Checkpoint 1 (b577b69, short prompts, ms/token): dense int8 8.4 | lm_head 2.9 | GPU experts 2.4 | other 2.4 |
GPU idle (CPU wait + gaps) ~4.2 (CPU work 5.0) | total 20.3 = 49.2 tok/s raw, 58.5 with MTP.

Parked (small lever: ~+0.6 tok/s per 100 slots): lazy KV (grow in chunks; ~70 MB = ~40 slots at 1K; the decode graphs
capture the KV pointers, so growth needs a re-capture), a dynamic expert cache (shrink as the context grows), a smaller
driver reserve (150 MB worked here; risky with other apps using VRAM).

### Checkpoint 2: int4 dense - not shipped (no tensor passes)
GEMV is ready: int4 gs64 lm_head 1.66 ms = 172.6 GB/s (92%; int8 2.77 ms). Quality is the blocker. Ideal saving if every
dense tensor were int4: ~804 MB, ~4.8 ms/token (dn_in 318 MB / 1.89 ms, lm_head 213 / 1.27, dn_out 105 / 0.63,
attn_qkv 79 / 0.47, attn_o 35 / 0.21, sh_gu 35 / 0.21, sh_d 18 / 0.11).
`--check --no-prefill`, one tensor type int4 at a time (int8 baseline mean KL 2.09e-3, top-1 98.1%); budget for the
whole mix: KL +1e-3, top-1 -0.5:

| tensor | RTN (with prefill) | MSE clip search | Hadamard 256 | Hadamard 1024 |
|---|---|---|---|---|
| dn_in | 3.58e-2 / 88.9% | 9.83e-3 / 97.2% | 9.14e-3 / 99.1% | 2.57e-2 / 97.2% |
| lm_head | 6.70e-3 / 95.8% | 7.83e-3 / 98.1% | 7.95e-3 / 96.3% | 7.54e-3 / 97.2% |
| dn_out | 6.01e-3 / 97.2% | 1.55e-2 / 96.3% | 1.12e-2 / 94.4% | 1.66e-2 / 96.3% |
| attn_qkv | (with attn_o) 2.39e-2 / 98.6% | 1.19e-2 / 97.2% | 1.21e-2 / 95.3% | 6.02e-3 / 98.1% |
| attn_o | | 7.49e-3 / 98.1% | 8.33e-3 / 97.2% | 1.91e-2 / 98.1% |
| sh_gu | (with sh_d) 7.33e-3 / 95.8% | 3.53e-3 / 96.3% | 4.29e-3 / 98.1% | 3.98e-3 / 97.2% |
| sh_d | | 5.70e-3 / 95.3% | 2.99e-3 / 97.2% | 4.36e-3 / 97.2% |

(dn_in also: Hadamard 64 1.24e-2 / 95.3%, 64 + MSE 1.26e-2, 256 + MSE 1.97e-2.) The closest, sh_d with Hadamard 256, adds
+0.9e-3 but drops top-1 0.9 points. Stopped after two rotation tries. Not tried: GPTQ / AWQ with calibration data,
6-bit dense with group scales (~0.78x the bytes of int8 per-row, ~2.4 ms/token if every tensor takes it).

## State at the context compaction (2026-10-19, branch `speed`)

Commits on `speed` since f28d388 (oldest first): 5e12c13 profiling; 5fab266 router/shared gate one launch; a4ecff6
streamer thread; 8dd7290 int8 GEMM 18.8 TFLOPS; 71ca6f9 tensor-core GQA prefill attention; 5bc3f70 non-temporal
copy; 3be0336 streamed-bytes counters; 83103d8 predicted stream-ahead; d6b63e9 CPU takes <= 2-row experts (prefill);
3b4117c GQA decode attention + MTP-round splits; 7b85298 D1 verification; b9fd776 D2 re-profile; b577b69 CPU experts
int8 + AVX-VNNI (decode + MTP round; new greedy baseline, binary build/moex_base.exe); d8909ab VRAM audit + Checkpoint 1;
fc649fe int4 dense measured, not shipped.

Where it stands (short prompts, ~1K context): 49.2 tok/s raw, ~58.5 with MTP (`moex bench` 49.9 / 53.5).
Prefill 4K ~763 tok/s, 16K ~710. Decode at 26K: 42 raw / 45 MTP.

### In the working tree, NOT committed (resume here)
- `MOEX_FQ=group:bits:gs:asym,...` (research): wmat keeps that group as F16 holding fake-quantized values
  (fakequant_rows_k in kernels.cuh): any format's quality end to end without a kernel.
- `MOEX_CHECK_DUMP=1 moex --check DIR --f16`: writes the run's logits as DIR's reference. tools/make_check_set.py
  builds check sets from a prompt file (chat template, `### assistant` / `### tool` turns) + a greedy continuation.
  build/ref_code1 (code edit, 713-token prompt) and build/ref_agent1 (tool-use trace, 221) = 129 positions each, F16
  reference (gitignored: 130 MB each). build/fq3.sh runs eval1 + code1 + agent1 per MOEX_FQ config.
- Real int6 kernels (`--q6 groups`): low-4-bit plane in q4's nibble layout + 2-bit high plane, gs64 fp32 scales,
  0.81x int8 bytes. gemv_q6_k (1 or 2 tokens: the MTP round reads W once), gemm_i8_k<true> for prefill,
  quantize_rows_q6_k (same values as fakequant 6-bit). test_kernels K11b: ALL PASS vs F16 on the same values;
  in_proj 0.145 -> 0.113 ms, out_proj 0.078 -> 0.066, lm_head 2.77 -> 2.36 (175 GB/s). Not yet default, not committed.

### 6-bit screening (fake-quant, three sets: eval1 + code1 + agent1 = 330 positions)
Noise: int8 baseline twice: mean KL 1.82e-3 / 2.09e-3 (agent1 3.44 / 4.19e-3), top-1 98.8% (4/330) both: ~+-0.15e-3.
One tensor type 6-bit gs64 at a time (KL added vs ~1.96e-3; top-1):
lm_head 2.32e-3 (+0.36) 99.1% | dn_out 2.21e-3 (+0.25) 99.7% | attn_qkv 2.36e-3 (+0.40) 99.7% | attn_o 2.94e-3 (+0.98)
99.1% | sh_gu 2.64e-3 (+0.68) 98.5% | dn_in 3.32e-3 (+1.36) 98.5% | sh_d: pending.
eval1 alone was misleading (dn_out +2.65e-3 there): 72 positions, a few dominate. Next: the mix lm_head + dn_out +
attn_qkv (+ sh_d) end to end on the three sets (budget +1e-3, -0.5 top-1, judged against the noise), then make it
the default with the real kernels and run the full gates (all three sets, test_kernels, A/B MTP on/off at 1K, prefill
output match), commit. Expected saving: ~0.19 x (lm_head 486 + dn_out 240 + attn_qkv 180 MB) at ~176 GB/s ~ 1 ms/token.

### Checkpoint 3 plan (next, before any more int4 work)
1. nsys at 1K decode: every kernel per token with time, bytes, achieved GB/s; flag those under ~60% of 188 GB/s
   (small GEMVs are latency-bound: out_proj 57%, in_proj 61%, shared down 32%).
2. Fuse small GEMVs sharing an input (attn q/k/v + gate already one matrix; router + shared-expert gate; DeltaNet
   in-projections already one; consider split-K / more blocks so small matrices fill all 20 SMs).
3. Fuse norm + residual (+ routing export) and small elementwise kernels (add_k, silu_mul, gated_add, table_k...).
4. CUDA graph for the whole decode step (today one graph per layer, 40 launches + host work between); report
   launches/token before and after.
5. CPU/GPU overlap: start CPU experts right after the router (shared expert + GPU experts meanwhile); GPU idle
   wait -> ~0. P-core pinning of the CPU expert threads (C11), with/without E-cores.
6. Re-measure; only if dense GEMVs are then bandwidth-bound, the int4 methods (asym g32, AWQ, GPTQ, int5),
   time-boxed: GPTQ stops if not >= 2 tensor types under budget after two tries.

### Parked, with reasons
- CH 2048 prefill (design above): not built; prefill target met for now.
- Lazy KV / dynamic expert cache / smaller driver reserve: +100 slots = +1.2% (below noise).
- int4 dense RTN / MSE clip / Hadamard 256 & 1024: no tensor within budget (table above).
- D3 variant hunt (int8 x vs int8 down input): skipped; D3 accepted on KL (+0.26e-3) and top-1.
- Opt-in --fast miss skipping, n-gram + MTP hybrid drafts, frequency-ranked draft vocab: not started (later checkpoint).
- Agent state snapshots (original step 2): not started.
- Possible issue: stopping a background shell here does not stop its moex children; check for stray moex/bash
  processes (and VRAM in use) before every benchmark.

## Speed pass 2 (branch `speed2`, 2026-10-19)

GPU runs go through the shared job queue (`C:\Users\neera\OneDrive\Desktop\gpu-queue\gpuq.py`): one job at a time,
an A/B or a gate is one script = one job. Earlier runs that overlapped another agent's moex were discarded.
Power: plugged in, Windows power mode "Balanced" (not "Best performance"): absolute tok/s may be a little low,
A/B comparisons are alternating in the same job.

### Step 1: 6-bit dense mix (a64c138)
Three check sets (eval1 + code1 + agent1, 330 positions; `build/fq3.sh`), int8 baseline 1.85-2.11e-3 (noise ~+-0.15e-3):

| mix | KL | added | top-1 |
|---|---|---|---|
| int8 (3 runs) | 2.11 / 1.85 / 2.05e-3 | - | 98.8% |
| fake-quant dn_out + lm_head + sh_d + attn_qkv | 3.02e-3 | +1.04e-3 (over) | 99.4% |
| fake-quant dn_out + lm_head + sh_d (2 runs) | 2.71 / 2.65e-3 | +0.70e-3 | 99.7% |
| **real gemv_q6, dn_out + lm_head + sh_d** (same job as int8 2.05e-3) | 2.51e-3 | **+0.46e-3** | 99.7% |

Default now `--q6 dn_out,lm_head,sh_d` (`--q6 none` = int8). lm_head 486 -> 394 MB; expert slots 1586 -> 1707 (ctx 2048).
Prefill vs token-by-token (4K prompt, 48 tokens): identical up to the known near-tie at token 16 (two branches,
int8 produces both run to run). A/B (`build/gate_s1.sh`, 1K prompt + 10 short prompts, 128 tokens, alternating):

| tok/s | int8 (b577b69) | int6 mix |
|---|---|---|
| raw | 48.1 / 48.3 | **50.3 / 50.0** |
| MTP | 57.1 / 58.4 | **59.2 / 60.3** |

## agent2 (branch `agent2`): n-gram drafts, deterministic mode, GPTQ / AWQ research

### A1: prompt-lookup n-gram drafts in the MTP round (opt-in: `MOEX_NGRAM=2`, needs the MTP head)
What: when the context's last 2..4 tokens occurred earlier in it (prompt + generated), the round drafts up to
`MOEX_NGRAM_MAX` (default 4, max 8) tokens continuing the earlier occurrence instead of the head's one; no match: the
head's draft as before. The head still runs every round (its own KV rows). Draft length adapts: alpha = EMA of the
per-token acceptance, length = 1 + alpha / (1 - alpha) within [1, max]. Verification is exact speculative sampling
(src/spec.h `spec_verify`: accept d with min(1, p(d) / q(d)), else draw from norm(max(0, p - q)); greedy: p and q
one-hot = keep a draft iff it is its row's argmax); sampled drafts are drawn from the continuation counts. The round
is K + 1 rows (VT 4 -> 9): the DeltaNet kernels snapshot the state after every row but the last (~66 MB per draft
token, allocated only for the n-gram max: 4 drafts = 111 fewer expert slots), rollback copies row a's state; int8
dense GEMV over up to 8 rows reads W once (`gemm` S <= 8, each row summed exactly as decode's GEMV); CPU pair kernel up
to 9 rows; session/serve and `--check` take rounds of any length; `--check` with `MOEX_NGRAM` teacher-forces
`MOEX_NGRAM_MAX` drafts per round.
Tests: tests/test_spec.cpp (CPU) ALL PASS: drafter, chains, adaptive length, p/q over 5 draft distributions (TV(out,
p) <= 0.0021, acceptance = sum min(p, q) +- 0.0008), a 3-draft round over a Markov target (TV of the first two tokens
0.0025 fixed / 0.0024 sampled drafts). test_kernels ALL PASS. `--check` (MTP on): eval1 1.53e-3 / 98.6% (as before);
code1 1 draft 3.87e-4 / 100%, 4 drafts teacher-forced 3.73e-4 / 100% (rows 1..4 and the rollbacks checked).
Greedy (MOEX_DETERMINISTIC=1, prompts of eval1 / code1 / agent1, 128 tokens): code1 and agent1 identical to plain decode
with MTP and with MTP + n-gram; eval1: both split from plain at token 61 (the verify rows' expert kernels round
differently from decode's, as plain MTP already did), n-gram vs MTP at token 100. So "identical to plain decode"
holds except at near-ties, exactly as for MTP today.
Tokens per round (greedy): code1 1.98 (head) -> **3.46** (n-gram drafts in 37% of rounds, 3.05 drafts, 73.8% kept);
eval1 1.83 -> 1.78 (7% of rounds); agent1 1.88 -> 1.84 (30%, 68%). Sampled (temp 0.7, one correctness run in the queue,
not a benchmark): 52.1 vs 52.4 tok/s overall, code1 60.1 vs 55.5. Speed needs the GPU owner's A/B.
Merge: independent of A2 / A3; off by default. Default-path changes: the MTP accept goes through spec_verify (greedy
unchanged; sampled: same distribution, other random draws for a seed); `gemm` int8 with 3..8 rows uses the GEMV
(prefill tails too); VT 4 -> 9 (scratch +~1.6 MB, lg2 +5 MB). Open: int6 matrices (speed2's default for dn_out,
lm_head, sh_d) have gemv_q6 for 1-2 rows only, so 3+ row rounds use the int6 tensor-core GEMM (fp16 activations).

### A2: deterministic test mode (`MOEX_DETERMINISTIC=1`)
Finding: the reductions are already in a fixed order. No float atomics on the GPU (the one atomicAdd compacts sampling
candidates, which are then sorted with an id tie-break); attention splits combine in split order; pairs_reduce sums
ranks in order; each CPU output element is one work item's fixed loop (no cross-thread sums). The run-to-run splits
come from residency: whether an admission copy has landed by the next layer decides if an expert runs on the GPU or
on the CPU (int8 activations, other rounding). In this mode poll_admissions waits until the loader has enqueued every
job and the copy stream is idle, so residency (and every CPU/GPU split, prefill's too) depends only on the tokens. In
this mode it also closes a window in claim_prefill_slots (the loader may have taken a job but not yet queued its copy).
Shown in the `[load]` line. Not for the low-RAM mode (own RAM cache reads are timing-dependent).
Test: `--check ref/eval1` twice with the logits dumped (MOEX_CHECK_DUMP), plain decode and MTP rounds: 72/72
positions bit-identical in both. Cost: decode ~2.5 tok/s (every admission waited for at every layer): gates only; the
default path is unchanged.
Merge: independent; touches poll_admissions and the [load] line only.

### A3: offline GPTQ / AWQ for the dense tensors (research, nothing ships from it)
Pieces: `MOEX_CALIB=dir` (+ `MOEX_CALIB_EVERY=n`): prefill dumps every dense matrix's input rows (fp16, one file per
group and layer); tools/calib_prompts.py: 200 calibration prompts (data/prompts.txt's 80 chat + 120 code windows of
this repo under 8 instructions; the check sets and h2h are left out); tools/quant_dense.py: rtn, awq (per input
channel s = rms^alpha, alpha grid on the output error tr(E H E^T), then per-group clip search; stored with s), gptq
(static groups, `--act-order`, `--clip`), awq+gptq; int4 / int6, gs 32 / 64, `--asym` (zero points); output in the
packed planes (q4 nibbles / q6 low + high planes, f32 scales, optional zeros and AWQ scales: src/dense_ovr.h).
`MOEX_DENSE=dir` loads such a file (kept as F16 holding the exact dequantized values, so any variant screens without a
kernel); tools/screen_dense.py --grid runs the per-tensor table (MOEX_FQ RTN column + MOEX_DENSE variants, eval1 /
code1 / agent1, KL added and top-1 vs the int8 baseline).
Tests: quant_dense.py --selftest (synthetic, int4 rel output err: rtn 1.25e-2, awq 7.1e-3, gptq 8.6e-3, gptq act-order
8.8e-3, gptq clip 5.4e-3, awq+gptq 6.3e-3, gptq gs32 6.6e-3, gptq asym 5.8e-3); tests/test_dense.cpp: the C++ decoder
gives Python's dequantized values bit for bit in 4 layouts (int4/int6, gs 64/32, sym/asym, with/without AWQ scales).
Calibration dump done (queue job, 244 s): 200 prompts, every 12th prompt token, 161 files, 2.8 GB.
Table: pending (queued after the n-gram follow-ups). Run: tools/screen_dense.py --grid (see its docstring).
Merge: research tooling; the engine hooks are opt-in env switches (no effect unset).

### A1b: n-gram follow-ups; n-gram parked (opt-in `MOEX_NGRAM=3`, off by default)
- Multi-row dense GEMV: int6 3..8 rows (x in shared memory up to 96 KB, else from global), int8 2..8 rows; test_kernels
  checks every row equals the 1-row GEMV exactly (2, 3, 5, 8 rows; int8 in_proj / out_proj / lm_head, int6 in_proj /
  out_proj / shared down / lm_head). 3 rows, int6, vs the int6 tensor-core GEMM they used before: lm_head 5.94 ->
  **2.56 ms** (162 GB/s), in_proj 0.36 -> 0.13, out_proj 0.19 -> 0.10, shared down 0.039 -> 0.020. The 2-row MTP round was
  already on GEMVs (gemv_i8 NT 2, gemv_q6<2>): lm_head 2.47 ms per round vs 2.36 for one row.
- Rollback by replay: one full DeltaNet state copy (after row 0, as plain MTP) + per-row conv ring (96 KB/layer) +
  rows 1..K-1's recurrence inputs (~1 MB per draft); a rollback past row 0 copies the row-0 state back and replays
  rows 1..a with dn_recur_k (same kernel, same order: bit-identical state). Rollback state with 4 drafts: 263 -> 74 MB
  (expert slots 1617 vs 1624 MTP-only; was -111). Cost: rollback to row 0 0.92-0.98 ms (copy), past row 0 1.30-1.40 ms
  (copy + replay of 1-3 rows), 3-4 per 384 tokens. `--check` with `MOEX_CHECK_WRONG=j` (draft j wrong every round, so
  every round replays j rows): code1 j=2 KL 1.58e-4, j=3 1.97e-4, top-1 100% (normal 4-draft 2.36e-4).
- Pick (token choice per verified row) timed apart from lm_head: greedy 0.04 ms/row, sampled (t 0.7, top-p 0.95)
  0.16-0.17 ms/row = 0.3 ms per round: under the 0.5 ms bar, sampling left as is.
- Profiler: the MTP-round expert counters now cover rounds of any length (they counted 2-row rounds only).
- Gating (DraftGate, src/spec.h): per prompt, n-gram drafts only while their tokens per ms of round beat the head's
  (rounds that captured a graph are excluded; the losing side is still tried every 4th round). Gated vs MTP-only, 2
  alternating runs, 128 tokens per prompt, greedy / sampled t 0.7 (tok/s):

  | prompt | greedy MTP-only | greedy gated | sampled MTP-only | sampled gated |
  |---|---|---|---|---|
  | chat (eval1) | 52.7 / 55.5 | 55.0 / 54.5 (+1.1%) | 54.1 / 56.6 | 57.1 / 54.3 (+0.6%) |
  | code (code1) | 62.2 / 62.3 | 59.3 / 60.5 (-3.7%) | 61.1 / 58.0 | 53.8 / 64.3 (-0.9%) |
  | agent (agent1) | 56.9 / 57.8 | 53.8 / 59.0 (-1.6%) | 54.7 / 55.2 | 55.1 / 57.0 (+1.9%) |
  | overall | 57.0 / 58.4 | 55.9 / 57.9 (-1.4%) | 56.5 / 56.6 | 55.3 / 58.2 (+0.4%) |

  (that run probed every 16th round; the gate found n-gram 19% ahead on code, 0.072 vs 0.061 tokens/ms, only after
  ~44 gated rounds.) Verdict: parked. On code the head already keeps 98% of its drafts (1.98 tokens per round), and
  run-to-run noise (53.8 vs 64.3 on one config) is larger than the effect. Kept as an opt-in: the verify path,
  sampling and rollback are exact and tested.
