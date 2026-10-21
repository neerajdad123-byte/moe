# moex vs llama.cpp vs ik_llama.cpp: Qwen3.6-35B-A3B on a 6 GB laptop

Same laptop, plugged in, Windows "High performance" plan, nothing else running (display on the Intel iGPU):
RTX 4050 Laptop 6 GB (driver 610.74), i7-13650HX (6P + 8E cores, 20 threads), 24 GB DDR5-4800 dual channel.
Measured 2026-10-20 (llama engines) and 2026-10-21 (moex at speed2 df0db07: expert cache sized from NVML's free VRAM).

## Results

Same prompts for every engine: 4 prompts of 1030 tokens and 2 prompts of 8030 tokens (slices of this repo's text +
"summarize, list three details", Qwen chat template, sent as the same token ids), 128 output tokens, greedy.
Each engine runs as its own OpenAI-compatible server with an 8K+ context, and one client times all of them the same way.

| engine | quant | decode 1K | decode 8K | prefill 1K | prefill 8K | VRAM | RAM working set / private |
|---|---|---|---|---|---|---|---|
| **moex** (default: MTP on) | int4 gs64 experts, int8 trunk | **67.8** (64.5-70.6) | **64.5** | **733** | **764** | 5.6 GB | 18.3 / 6.2 GB |
| **moex --fast** | same | **69.2** (63.2-73.5) | **67.5** | 725 | 765 | 5.6 GB | 18.3 / 6.3 GB |
| ik_llama.cpp 565f08d1 | Q4_K_M GGUF | 36.0 (35.8-36.2) | 35.1 | 333 | 336 | 5.9 GB | 16.5 / 23.2 GB |
| llama.cpp b11756 | Q4_K_M GGUF | 33.3 (32.4-34.0) | 32.8 | 202 | 189 | 5.9 GB | 17.9 / 10.0 GB |

All figures are tok/s unless marked otherwise. Decode = (tokens - 1) / (time from the first streamed text to the
end), the mean over prompts with min-max in brackets. Prefill = prompt tokens / time to the first streamed text.
VRAM = peak `nvidia-smi` memory.used while serving. RAM = the server process's peak working set and private bytes.
- **moex and the 6 GB card:** moex fills the free VRAM (as NVML / nvidia-smi count it) with expert slots, minus a
  320 MB margin. Before df0db07 it used CUDA's count, which on Windows leaves out ~1 GB: 4.9 GB, 65.5 / 63.0 tok/s.
  The llama engines' 5.9 GB is what the hand-tuned `--n-cpu-moe` asks for, not a computed limit.
- **RAM:** the llama engines memory-map the GGUF, so their working set includes file pages. ik_llama.cpp also copies
  the CPU experts into private memory. moex locks the expert file pages in RAM.
- **Every prompt produced 128 tokens on every engine.** The llama engines get `ignore_eos`. moex has no such option,
  but none of its replies hit end-of-text early.

moex vs the fastest llama engine: decode **1.9x** at 1K, **1.8x** at 8K (default), prefill **2.2-2.3x**.

**Quants:** the closest pair available. moex runs its own container: int4 group-64 experts with int8 dense weights,
int6 for dn_out / lm_head / shared-expert down, plus the MTP head. llama.cpp and ik_llama.cpp run the official
ggml-org `Qwen3.6-35B-A3B-Q4_K_M.gguf` (19.0 GiB): Q4_K experts, about 4.5 bits per weight. Both are ~4.5-bit expert
weights with an 8-bit-class trunk. moex's own quality check against the f16 reference is in PLAN.md
(MTP: mean KL 2.6e-3, top-1 99.4%).

**KTransformers: skipped**, decided within ~5 minutes, no install attempted. Its PyPI wheels (`ktransformers` /
`sglang-kt` 0.7.2) are Linux-only (`manylinux_2_35_x86_64`). `kt-kernel` lists only Linux and macOS, and serving now
goes through SGLang, which has no native Windows build. Its native Windows support (Aug 2024) belongs to the archived
old framework, which predates Qwen3.6's hybrid DeltaNet layers. Running it here would mean WSL2 plus its own model
conversion, well past the 30-minute cap.

Short test on purpose (4 + 2 prompts). The longer 16-prompt moex runs are in PLAN.md and agree within a few %.

## Best flags per engine (how they were picked)

**llama.cpp b11756** (official `llama-b11756-bin-win-cuda-13.4-x64.zip` + cudart 13.4), tuned with `llama-bench`:
- `--n-cpu-moe`: 32 is the fewest CPU-expert layers that fits 6 GB: 32.8 tok/s decode. At 31 it spills to shared
  memory: 15 tok/s.
- `-t 14` beat 6 / 8 / 10 / 20: 33.6 tok/s.
- `-ub 2048 -b 2048`: same prefill as 512 but steadier.
- `-fa on`.

**ik_llama.cpp 565f08d1**, built from source: CUDA 13.3, sm_89, `GGML_NATIVE=ON`, Ninja, MSVC.
- `--n-cpu-moe 33`: 32 decodes the same but drops prefill to 238.
- `-t 14`: 10 and 20 were slower.
- `-fa 1`, with ik's default fused MoE (`-fmoe 1`).
- None of `-rtr 1`, `-muge 1`, `-ger 1` or `-ub 2048` helped (each within noise or slower).

**moex**: no flags beyond the model path and context. MTP is on by default in `serve`, CPU threads and expert slots
are picked automatically. `--fast` is the opt-in approximate mode (low-ranked CPU experts finish a layer late:
KL 3.8e-3 vs 2.5e-3).

## Exact commands

Servers (one at a time, port 8091: 8080 is taken by a Windows service on this PC):

```bash
G=C:/models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
# moex default / --fast
build/moex.exe serve --model C:/models/qwen36_i4_gs64 --ctx 9216 --port 8091 --load
build/moex.exe serve --model C:/models/qwen36_i4_gs64 --ctx 9216 --port 8091 --load --fast
# llama.cpp b11756
llama-server.exe -m $G -ngl 99 -fa on --n-cpu-moe 32 -t 14 -ub 2048 -b 2048 -c 9216 -np 1 --port 8091
# ik_llama.cpp 565f08d1
llama-server.exe -m $G -ngl 99 -fa 1 --n-cpu-moe 33 -t 14 -c 9216 -np 1 --port 8091
```

Prompts and client, once a server answers `GET /health` with 200:

```bash
python tools/mk_long.py 1000 16 build/long1k.txt   # 16 prompts of ~1K tokens (the first 4 are used)
python tools/mk_long.py 8000 16 build/long8k.txt   # 16 prompts of ~8K tokens (the first 2 are used)
python tools/compare.py --url http://127.0.0.1:8091 --prompts build/long1k.txt --count 4 --n 128 --pid <server pid> --label X
python tools/compare.py --url http://127.0.0.1:8091 --prompts build/long8k.txt --count 2 --n 128 --pid <server pid> --label X
```

`tools/compare.py` sends `POST /v1/completions` with the token ids, `temperature 0, top_k 1, max_tokens 128,
ignore_eos, cache_prompt false`, streamed. It sends one warm-up request first (not counted) and samples VRAM and the
server's RAM every 0.5 s. The prompt builder is the same as `tools/bench.py`'s: slices of the repo's tracked text
files at evenly spread offsets.

Tuning sweeps:

```bash
llama-bench.exe -m $G -ngl 99 -fa on -ncmoe 36,34,33,32,31,30 -p 1024 -n 64 -t 8
llama-bench.exe -m $G -ngl 99 -fa on -ncmoe 32 -p 0 -n 64 -t 6,8,10,14,20
llama-bench.exe -m $G -ngl 99 -fa on -ncmoe 32,33 -p 1024 -n 0 -ub 512,1024,2048 -b 2048 -t 8
# ik_llama.cpp (one value per run)
llama-bench.exe -m $G -ngl 99 -fa 1 --n-cpu-moe {33,32,31,30} -p 1024 -n 64 -t 14
llama-bench.exe -m $G -ngl 99 -fa 1 --n-cpu-moe 33 -p 1024 -n 64 {-t 10 | -t 20 | -t 14 -rtr 1 | -t 14 -muge 1 | -t 14 -ub 2048 -b 2048 | -t 14 -ger 1}
```
