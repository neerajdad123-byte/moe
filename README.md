# MoEx

Experimental CUDA inference runtime for `Qwen3-30B-A3B` GGUF models on a 6 GB GPU. The project keeps static weights and selected MoE experts resident in VRAM, then measures a real cold discovery/pin pass followed by repeatable warm decode passes.

## Current verified ceiling

The latest short-context `"hi"` benchmark uses a real model load, discovers and pins the experts used by `"hi"` plus four generated tokens, then replays the prompt three times with fresh KV cache and expert uploads disabled.

Branch `feat/40toks-kernel-opts` (warp-shuffle GQA, `gpu_dispatch`, CUDA graph capture/replay):

| Metric | Result |
|---|---:|
| Warm decode (best pass) | 18.15 tok/s |
| Warm decode (3-pass avg) | 16.73 tok/s |
| Warm TTFT (avg) | ~64 ms |
| Warm expert hit rate | 100% |
| Warm H2D transfers | 0 |

CUDA graphs (llama.cpp technique) are enabled on Phase 1 and are correct, but only move the needle ~2–3% on this 4050/WDDM box — the decode path is compute-bound, not launch-bound. Global Q8/`dp4a` MMVQ (also from llama.cpp) is implemented but **not** on the default path yet: fused Q8×Q4_K expert kernels still hit illegal memory access during bring-up; float GEMV remains the stable ceiling.

Path to 40+ still needs a proven MMVQ/`dp4a` win (≤25 ms/token at 100% hit, 0 H2D).

The accepted CUDA changes currently include fused expert gate/up/SiLU, fused expert down/residual, fused QKV, fused attention output/residual, fused Q/K RMSNorm/RoPE, warp-shuffle GQA, GPU dispatch, device-side decode args, and CUDA graph replay.

Detailed real-pass logs and the per-token trace are under `build/`.

## Build

Requirements:

- Visual Studio C++ toolchain
- CUDA toolkit targeting `sm_89`
- A GGUF model at `C:\models\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf`, or another path passed on the command line

Build the generator:

```bat
build\bgen.bat
```

## Run the pinned ceiling benchmark

```powershell
.\build\moex_generate.exe C:\models\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf 4 3
```

Arguments are model path, generated tokens per pass, and warm replay count. Optional 4th arg `gpu_dispatch` (default 1) also enables CUDA graphs on Phase 1. The run writes `build/profile_trace.csv`, containing real per-token latency, phase timing, transfer totals, and VRAM usage.

## Repository layout

- `src/cuda/` — forward path, quantized GEMV, MMVQ helpers, CUDA kernels, profiler
- `src/model/`, `src/gguf/` — GGUF parsing, model manifest, tokenizer support
- `src/tools/` — generation, residency, and dequant validation tools
- `build/` — build scripts and selected benchmark evidence
- `2026-07-15-moex-unified-design.md` — design notes

## Benchmark discipline

Headline warm decode speed is measured with the in-process profiler disabled. A separate synchronized pass generates detailed profiling data, so instrumentation does not contaminate the rate being reported.
