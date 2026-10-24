# GGUF import work log (branch `gguf-import`)

Work done by Claude (session started from the main moex session) on 2026-10-23 and 2026-10-24, on the laptop
(RTX 4050 Laptop 6 GB, i7-13650HX, 25 GB RAM, Windows 11). Worktree: `C:\Users\neera\OneDrive\Desktop\moe-v2`,
branch `gguf-import`, pushed to `origin/gguf-import` (GitHub `neerajdad123-byte/moe`). **Not merged into main, not
released.** All GPU and heavy CPU work ran through the GPU queue (`C:\Users\neera\OneDrive\Desktop\gpu-queue\gpuq.py`).

## What it does now

`moex import FILE.gguf [--out DIR]` converts a GGUF of **Qwen3.5/3.6-35B-A3B or a finetune** (architecture
`qwen35moe`) once into moex's own model folder, which `moex chat / serve / bench --model DIR` then runs with every
feature (VRAM cache, RAM / SSD modes, prefetch, MTP, `--check`). `--model X.gguf` tells the user to import first.

- Reads GGUF v2/v3; checks every shape against the 35B-A3B model and refuses anything else by name.
- Reads 18 tensor types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS,
  IQ3_XXS, IQ3_S, MXFP4 (each decodes exactly like llama.cpp's reference, `gguf` Python package). IQ2_*/IQ1_* refused.
- Undoes llama.cpp's converter changes: RMSNorm weights stored as 1+w, DeltaNet value heads reordered, A_log stored
  as -exp(A_log).
- Experts go into a new **zero-point int4 format**: unsigned nibbles, `w = d * sc * u + dm * mn`, sc (uint8) and mn
  (int8) per 32 weights, (d, dm) fp16 per 256 = GGUF Q4_K's own layout. **Q4_K experts are copied bit for bit**
  (error 0). Other types are refit per 32 by least squares. 1.82 MB per expert (+2.8% vs the default model's format).
- Dense tensors: dequantized to F16 (moex quantizes them to int8/int6 at load, as for the default model).
- Writes `tokenizer.json` (vocab and merges identical to the default model's), `config.json`, the GGUF's
  `chat_template.jinja`, and `mtp.safetensors` when the GGUF has the MTP layer (nextn).
- Writes into `DIR.part` and renames at the end; prints the disk cost first (~23-24 GB).
- Chat templates: the GGUF's template is kept; moex follows Unsloth's variant (developer role, two system messages
  merged, Python-style tool arguments) and Ornith's (keeps past reasoning). Unknown templates fall back to Qwen3.6's.

## Timeline and commits

| Time (commit) | Commit | What |
|---|---|---|
| 2026-10-23 02:08 | `6ebce78` | `moex import`: GGUF reader, mapping, int4 requantizer, tokenizer, MTP, CPU tests |
| 2026-10-23 02:08 | `6d2a89b` | `tools/gguf_import_check.py`: imported folder vs the default model, tensor by tensor |
| 2026-10-23 02:47 | `23ce019` | `tools/llama_check.py`: llama.cpp on a GGUF vs the `--check` reference outputs |
| 2026-10-23 05:43 | `a21202d` | Engine: zero-point int4 experts on all 6 expert paths (GPU decode, GPU prefill GEMV and tensor cores, CPU decode, CPU prefill, MTP pairs) |
| 2026-10-23 17:33 | `f320b1b` | Import writes zero-point experts (Q4_K carried over) |
| 2026-10-24 07:08 | `d82b04b` | Chat-template variants + offline template check (`tests/test_template.cpp`, `check_template.py --offline`) |
| 2026-10-24 07:09 | `244f6a0` | CI: GGUF unit tests, import command-line cases, offline template check |
| 2026-10-24 07:14 | `6e92e46` | 11 more GGUF quant types (Q2_K ... MXFP4), IQ3 grids generated from gguf-py |
| 2026-10-24 07:15 | `b8bc693` | README: "Use a GGUF you already have" |
| 2026-10-24 07:24 | `62d0f9e` | `llama_check.py --dump`: llama.cpp's outputs as `--check` sets (test for finetunes) |
| 2026-10-24 07:48 | `0f452dd` | Compact zero-point layout (8-bit scales, Q4_K bits exact, experts fit in RAM again) |

Other events:
- 2026-10-23 ~01:40: the GPU queue's runner was not running (dead since 2026-10-21); started a new one. It died
  again twice when the laptop slept; restarted each time (`python gpuq.py run`, one instance; logs `runner8-10.log`).
- 2026-10-23: the user stopped a first import and an Ornith download (laptop busy); partial files deleted.
- 2026-10-23: the old symmetric import (21.4 GB) was deleted to free space once its comparison was done.
- 2026-10-24 07:18-08:54: Ornith-1.5 Q4_K_M downloaded (one connection reset at 3.5 GB, resumed).

## Results

All numbers: mean KL (lower = closer) and top-1 agreement, on the three `--check` sets (prose `ref/eval1` 72
positions, code `ref_code1` 129, agent `ref_agent1` 129). The reference outputs of these sets come from the default
(colibri int4) model itself, not from the original BF16 model.

### Against the default model's reference outputs

| | prose | code | agent |
|---|---|---|---|
| default model (colibri int4), moex | 3.53e-3, 100% | 2.78e-4, 100% | 4.61e-3, 97.7% |
| ggml-org Q4_K_M in llama.cpp b11756 | 2.00e-2, 93.1% | 1.18e-3, 100% | 2.21e-2, 92.2% |
| same GGUF, first moex import (symmetric int4, requantized) | 4.15e-2, 91.7% | 1.68e-3, 99.2% | 3.55e-2, 90.7% |
| same GGUF, moex zero-point import (Q4_K exact) | 2.64e-2, 91.7% | 7.42e-4, 99.2% | 2.66e-2, 92.2% |

The ~2e-2 of llama.cpp is the distance between two different 4-bit versions of the model (colibri vs the GGUF),
not an engine error.

### moex vs llama.cpp on the same GGUF (llama.cpp's outputs as reference, `llama_check.py --dump`)

| | prose | code | agent |
|---|---|---|---|
| moex, zero-point import, half2 layout (RAM cache capped at 4 GB) | 4.59e-3, 98.6% | 2.66e-4, 99.2% | 5.85e-3, 95.3% |
| moex, zero-point import, compact layout (all experts in RAM) | 5.10e-3, 97.2% | 2.66e-4, 99.2% | 5.36e-3, 95.3% |

Same order as moex vs its own reference on the default model: moex runs the GGUF as faithfully as llama.cpp.

### Speed (`moex bench`, laptop busier than in the release measurements)

| run (2026-10-24) | MTP off | MTP on | MTP on + --fast |
|---|---|---|---|
| zero-point import of ggml-org Q4_K_M (no MTP layer in that GGUF) | 44.9 | - | 47.7 (--fast) |
| default model, same job | 40.8 | 59.2 | 67.9 |

Earlier the same day the default model gave 58.2 / 65.5 / 55.4 with only 1.7-11 GB RAM free (user working).

### Ornith-1.5-35B-A3B Q4_K_M (finetune, MTP layer, Q4_K + Q6_K experts), 2026-10-24 18:54-19:02

- Import: 98 s, 24.4 GB folder, MTP head written. Worst expert error vs the GGUF 0.051 relative RMS (the Q6_K
  experts of 21 layers refit to 4 bits; Q4_K experts exact).
- llama.cpp on Ornith vs the default model's reference outputs: KL 0.49 / 0.087 / 0.28, top-1 77.8 / 99.2 / 86.0%
  (a different model, as expected for a finetune).
- moex vs llama.cpp on the same Ornith file (`llama_check.py --dump`):

| | prose | code | agent |
|---|---|---|---|
| moex default | 9.44e-3, 97.2% | 3.27e-3, 100% | 3.11e-2, 92.2% |
| moex --load --mtp (Ornith's own MTP head, imported) | 1.33e-2, 93.1% | 1.57e-3, 100% | 2.79e-2, 92.2% |

  Further from llama.cpp than on the ggml-org file (5e-3 / 3e-4 / 5e-3): Ornith's Q6_K experts (and Q6_K dense
  tensors) are rounded to 4 / 8 bits by moex, where llama.cpp keeps 6 bits.
- Speed (same job, idle laptop): Ornith MTP off 60.2, MTP on 68.8, MTP on + --fast 68.6 tok/s; default model
  61.5 / 73.4 / 76.2. The imported MTP head works (+14% with MTP).

### Unit tests (no GPU)

- `build\test_gguf.exe`: all 18 types decode exactly as gguf-py (max error 0), header parsing, truncated files
  refused, value-head un-permutation, half<->float, Q4_K experts exact (error 0), Q6_K -> 4 bits relative RMS 0.0737,
  least-squares refit never worse than min/max rounding (0.88x the squared error).
- `build\test_kernels.exe KZ`: zero-point experts on every path vs a float reference: GPU 5e-7, tensor cores 5e-4,
  CPU int8 activations 1.6-1.8e-2. K3/K4/PF1 (default format) unchanged.
- `tools/check_template.py --offline`: Qwen3.6's template (used unchanged by ggml-org, bartowski, Holo-3.1, HauhauCS,
  huihui, Qwen3.8-Distill, APEX builds), Unsloth's, Ornith-1.5's and Qwen3.5's: all cases token-identical.
  Tiel-Coder's "froggeric" template: different (own persona and tool text), falls back to Qwen3.6's format.

## GGUF files checked (headers only, 16-40 MB each)

Same shape as moex's model (qwen35moe, 40 layers or 41 with MTP, vocab 248320, pre-tokenizer qwen35):

| Repo / file | MTP | expert types | other types | template |
|---|---|---|---|---|
| ggml-org Qwen3.6 Q4_K_M (on disk) | no | Q4_K | Q8_0, Q4_K, Q6_K | Qwen3.6 |
| ornith-ai Ornith-1.5 Q4_K_M | yes | Q4_K, Q6_K | Q4_K, Q6_K | Ornith variant |
| peculiar-ragdoll Tiel-Coder MTP UD-Q4_K_XL / UD-Q4_K_S | yes | Q4_K, Q5_K, Q6_K, Q3_K | Q8_0 | froggeric (not matched) |
| peculiar-ragdoll Cyber-Tiel-Coder UD-Q4_K_M | no | Q4_K, Q5_K, Q6_K | Q8_0, Q6_K | own |
| empero-ai Qwen3.8-35B-A3B-Distill Q4_K_M | yes | Q4_K, Q6_K | Q4_K, Q6_K | Qwen3.6 |
| Jackrong Qwopus3.6-Coder MTP Q4_K_M | yes | Q4_K, Q6_K | Q4_K, Q6_K | own (py args) |
| Hcompany Holo-3.1 q4_k_m | no | Q4_K, Q6_K | Q4_K, Q6_K | Qwen3.6 |
| unsloth Qwen-AgentWorld UD-Q4_K_M | no | Q4_K, Q5_K, Q6_K | Q8_0, Q6_K | own |
| HauhauCS Qwen3.6 Uncensored Q4_K_P | no | Q4_K, Q6_K | Q4_K, Q6_K | Qwen3.6 |
| huihui Qwen3.6 abliterated MTP Q4_K | yes | Q4_K, Q6_K | Q4_K, Q6_K, F16 | Qwen3.6 |
| mudler Ornith-1.5 APEX-Balanced | no | Q5_K, Q6_K | Q6_K, Q8_0, Q5_K | Qwen3.6 |
| LuffyTheFox Genesis-Hermes MTP-APEX | yes | Q5_K, Q6_K, Q8_0 | Q6_K, Q8_0, BF16, F16 | Qwen3.6 |
| bartowski Qwen3.6 Q4_K_M | yes | Q4_K, Q6_K, Q8_0 | Q6_K, Q4_K, Q8_0, Q5_K, BF16 | Qwen3.6 |
| unsloth Qwen3.6 UD-IQ4_XS / UD-Q3_K_M | no | IQ4_XS, IQ3_S / IQ3_XXS, Q6_K | Q8_0, Q6_K | Unsloth variant |
| unsloth Qwen3.6 MXFP4_MOE | no | MXFP4, Q5_K, Q6_K | Q8_0 | Unsloth variant |
| unsloth Qwen3.5 Q4_K_M | no | Q4_K, Q5_K | Q8_0, Q6_K | Qwen3.5 |

Refused: openbmb MiniCPM-V-4.7-35B-A3B (vocabulary 248,144, not 248,320).

## Files on disk (2026-10-24 18:51)

| Path | Size | What | Keep? |
|---|---|---|---|
| `C:\models\qwen36_i4_gs64` | 24 GB | the default model (colibri int4 + MTP head) | yes |
| `C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf` | 20 GB (19.0 GiB) | ggml-org GGUF, downloaded 2026-10-12 | the test file |
| `moe-v2\build\llamaref_ornith\` | ~300 MB | llama.cpp's outputs on Ornith as `--check` sets | test data (not in git) |
| `C:\models\gguf\dl\ornith_curl.log`, `C:\models\gguf\curl.log` | <1 MB | download logs | can delete |
| `moe-v2\build\llamaref_qwen36\` | 314 MB | llama.cpp's outputs on the ggml-org GGUF as `--check` sets | test data (not in git) |
| `moe-v2\build\*.sh`, `*.bat` | small | gate scripts: `zp_gate.sh`, `z8_gate.sh`, `llama_ref_gate.sh`, `zp_test.bat` | not in git |

Free on C: 38 GB at 18:51, ~16 GB after the Ornith import at 18:55 (424 GB disk).
Deleted on the user's request at 2026-10-24 ~19:10: the Qwen3.6 `-zp` import (22 GB), the Ornith GGUF (21.7 GB) and
its import (23 GB). Free on C: afterwards: 79.7 GB.

Hand-over notes: `C:\Users\neera\AppData\Local\Claude-Data\claude_pzs8sxrjxfjjc\side-session-notes\local_4825b7a9-af6b-4b01-870b-59d11a1b098c\gguf-import-state.md`.
GPU queue logs: `C:\Users\neera\OneDrive\Desktop\gpu-queue\logs\` (job names `gguf-*`, `zp-*`, `z8-*`, `llamaref-*`, `tpl-*`).

## Code changed (on the branch)

| File | Change |
|---|---|
| `src/gguf.h` (new) | GGUF reader, 18 dequantizers, shape checks, tensor mapping, zero-point requantizer, tokenizer.json, import |
| `src/gguf_iq.h` (new, generated) | IQ3_XXS / IQ3_S grids and sign table from gguf-py (`tools/gen_iq_tables.py`) |
| `src/kernels.cuh` | zero-point expert format on decode, prefill GEMV and tensor-core prefill kernels (templated; default path unchanged) |
| `src/cpu_expert.h` | zero-point format on the CPU int8 decode, MTP pairs and fp32 prefill paths |
| `src/moex.cu` | `moex import`, `--out`, `.gguf` as `--model` message, expert format detection, slot sizes, template loading |
| `src/session.h` | chat-template variants (developer / merged system, Python-style tool args, keep reasoning) |
| `tests/test_gguf.cpp`, `tests/gguf_blocks.bin` (new) | CPU tests of the reader and the zero-point requantizer; real/random blocks with gguf-py's values |
| `tests/test_template.cpp` (new) | renders prompts without the model, for `check_template.py --offline` |
| `tests/test_kernels.cu` | KZ test |
| `tools/gguf_import_check.py`, `tools/llama_check.py`, `tools/make_gguf_fixture.py`, `tools/gen_iq_tables.py` (new) | checks and generators |
| `tools/check_template.py` | `--offline`, imported-template mode, variant cases |
| `build.bat`, `.github/workflows/ci.yml`, `README.md` | builds, CI steps, docs |

## Findings worth knowing

- Re-quantizing a GGUF into moex's symmetric int4 doubled the GGUF's own error (4.15e-2 vs 2.00e-2); keeping the
  GGUF's Q4_K values (zero-point format) removed that.
- The default model (colibri) is plain round-to-nearest int4 (scale = max/7, the -8 level never used), not
  calibrated. Calibrated int4 builds of Qwen3.6 exist (Intel AutoRound, palmfuture GPTQ: symmetric, groups of 128,
  convertible to moex without loss) and may be better; not measured. A fair comparison needs reference outputs from
  the BF16 model (~72 GB download).
- moex's low-RAM mode crashes (`CUDA out of memory` at cudaHostRegister) when RAM is tight while pinning its own
  expert cache. Not fixed (existing code; seen with the first zero-point layout).
- The GPU queue runner dies when the laptop sleeps; jobs then wait silently.
- The import took 146 s once and 778 s while the Ornith download ran at the same time.

## Not done yet

1. Ornith-1.5 is done (above). Its 6-bit experts lose accuracy when rounded to 4 bits (agent set KL 3.1e-2 vs
   llama.cpp); keeping Q5_K/Q6_K experts at more bits would close that, at more memory.
2. Speed of the ggml-org import on an idle laptop (Ornith's is above).
3. Converters for other downloads: FP8, GPTQ / AutoRound (lossless), AWQ (lossless into the zero-point format), NVFP4.
4. IQ2 / IQ1 GGUF types; Tiel-Coder's custom template; MiniCPM-V's different vocabulary.
5. Merge to main and release: the user's decision.
