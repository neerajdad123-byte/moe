# MoEx

Experimental CUDA inference runtime for `Qwen3-30B-A3B` GGUF models on a 6 GB GPU. The project keeps static weights and selected MoE experts resident in VRAM, then measures a real cold discovery/pin pass followed by repeatable warm decode passes.

## Current verified ceiling

The latest short-context `"hi"` benchmark uses a real model load, discovers and pins the experts used by `"hi"` plus four generated tokens, then replays the prompt three times with fresh KV cache and expert uploads disabled.

| Metric | Result |
|---|---:|
| Warm decode | 17.35 tok/s |
| Warm TTFT | 58.8 ms |
| Warm expert hit rate | 100% |
| Warm H2D transfers | 0 |
| Cold expert upload | 3.18 GiB |
| Cold expert upload time | 1.30 s |

The accepted CUDA changes currently include fused expert gate/up/SiLU, fused expert down/residual, fused QKV dispatch with per-tensor quantization support, fused attention output/residual, and fused Q/K RMSNorm/RoPE.

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

Arguments are model path, generated tokens per pass, and warm replay count. The run writes `build/profile_trace.csv`, containing real per-token latency, phase timing, transfer totals, and VRAM usage.

## Repository layout

- `src/cuda/` — forward path, quantized GEMV, CUDA kernels, profiler
- `src/model/`, `src/gguf/` — GGUF parsing, model manifest, tokenizer support
- `src/tools/` — generation, residency, and dequant validation tools
- `build/` — build scripts and selected benchmark evidence
- `2026-07-15-moex-unified-design.md` — design notes

## Benchmark discipline

Headline warm decode speed is measured with the in-process profiler disabled. A separate synchronized pass generates detailed profiling data, so instrumentation does not contaminate the rate being reported.
