<div align="center">

# moex

**Run Qwen3.6-35B-A3B at 68 tokens/s on a 6 GB laptop GPU.**<br>
A 35-billion-parameter model, ~1.9x faster than tuned llama.cpp and ik_llama.cpp, with nothing to install but the NVIDIA driver.<br>
<sub>moex with multi-token prediction (its default) vs llama.cpp / ik_llama.cpp without speculative decoding.</sub>

[![Release](https://img.shields.io/github/v/release/neerajdad123-byte/moe?label=download&color=2a78d6)](https://github.com/neerajdad123-byte/moe/releases/latest)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-555)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20RTX%2020%2B%20%C2%B7%206%20GB%2B-76b900)
[![License](https://img.shields.io/badge/license-MIT-555)](LICENSE)

<img src="docs/speed.svg" alt="Decode speed at 1K-token prompts: moex 67.8 tokens/s with multi-token prediction vs ik_llama.cpp 36.0 and llama.cpp 33.3 tokens/s (both hand-tuned, no speculative decoding), 1.9x faster" width="760">

</div>

## Get started in 3 steps

You need: Windows 10/11, an NVIDIA RTX 20 series or newer GPU with 6 GB+ VRAM, NVIDIA driver 580 or newer (older
driver? use the CUDA 12 build, see below), 22 GB of free disk and ideally 24 GB of RAM (16 GB works, more slowly).
No CUDA toolkit, no Python.

**1. Install moex.** Pick one:

- **Installer (recommended): `moex` works in every terminal, from any folder.** In PowerShell:

  ```powershell
  irm https://raw.githubusercontent.com/neerajdad123-byte/moe/main/install.ps1 | iex
  ```

  It puts `moex` (~20 MB) in `%LOCALAPPDATA%\Programs\moex` and on your PATH, no admin rights needed. Open a **new**
  terminal afterwards. Run the same line again to update.

- **Zip:** download [`moex-windows-x64.zip`](https://github.com/neerajdad123-byte/moe/releases/latest), unzip it
  anywhere, and open a terminal in that folder. There, PowerShell needs `.\` in front: `.\moex download`,
  `.\moex chat`. To use plain `moex` from anywhere later, run `.\moex install` once. On a driver older than 580
  (check with `nvidia-smi`), take `moex-windows-x64-cuda12.zip` instead (the installer does this by itself).

**2. Download the model.** In any terminal:

```bat
moex download
```

This fetches the model (22 GB, from Hugging Face) and the MTP head (0.5 GB, see `--no-mtp`) into `models\` next to
`moex.exe`. It resumes where it left off: if your connection drops, run it again.

**3. Chat.**

```bat
moex chat --load
```

```text
[before load] RAM 18.8 GB free of 25.5 GB | VRAM 5.3 GB free of 6.4 GB
[load] 1496 most-used experts preloaded into VRAM
[after load] RAM 0.3 GB free of 25.5 GB | VRAM 0.3 GB free of 6.4 GB  (by design: free VRAM became expert slots, RAM holds the experts)

Now the stage is yours.

> Explain in 3 sentences why the sky is blue.
Sunlight reaches Earth's atmosphere and is scattered in all directions by the gases and particles in the air. Blue
light is scattered more than other colors because it travels in shorter, smaller waves. This phenomenon, known as
Rayleigh scattering, is why we see a blue sky most of the time.
[60 tokens, 44.5 tok/s, first token 0.43 s, context 84]
```

`--load` takes about 30 seconds, then every reply runs at full speed. Without it the prompt appears in ~7 seconds, but
the first replies are slower while the model finishes loading into RAM. RAM and VRAM look full after loading on
purpose: free VRAM becomes expert cache, and the experts stay in RAM so the CPU never waits on the disk.

<details>
<summary><b>Other ways to get the model</b></summary>

The model is [Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64)
on Hugging Face. You can also download it with the Hugging Face CLI, or file by file from the website, into any folder:

```bat
hf download Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64 --local-dir D:\models\qwen36
moex chat --load --model D:\models\qwen36
```

`--model DIR` (or the `MOEX_MODEL` environment variable) tells moex where the model is. `moex download --model DIR`
downloads to that folder instead.

</details>

## Commands

| Command | What it does |
|---|---|
| `moex download` | download or resume the model |
| `moex chat` | chat in the terminal |
| `moex serve` | OpenAI-compatible API for other apps |
| `moex bench` | measure speed on your PC (~2-4 min, one model load): MTP off, MTP on and --fast tok/s, your PC's specs, a line to share |
| `moex install` | put moex on your PATH (for zip users) |
| `moex help`, `-h`, `--help` | all commands and options |
| `moex --version`, `-v` | the version |

Uninstall: delete `%LOCALAPPDATA%\Programs\moex` (and your model folder), and remove that folder from your PATH
(Windows search: "Edit environment variables for your account").

## Chat options

Type a message and press Enter. `/reset` starts a new conversation, `/exit` quits. Speed is shown after every reply.

| Option | What it does |
|---|---|
| `--load` | preload your most-used experts into VRAM and all experts into RAM before the prompt |
| `--no-mtp` | turn off multi-token prediction (the model drafts the next tokens and checks them in one pass: ~35% faster on the test laptop, on by default when your RAM holds every expert) |
| `--fast` | 2-9% faster, slightly approximate: the least important CPU experts finish one layer late (mean KL 3.1e-3 instead of 2.6e-3, see Accuracy). Off by default |
| `--system "You are ..."` | system prompt |
| `--think` | let the model reason step by step before it answers |
| `--temp 0.7 --top-p 0.8 --top-k 20` | sampling (these are the defaults; `--temp 0` always picks the likeliest token) |
| `--seed N` | the same answer every time for the same input |
| `--ctx 8192` | context length in tokens; long contexts work (26,500 tested), each 1,000 tokens costs ~10 MB of VRAM |
| `--model DIR` | model folder (default: `models\` next to `moex.exe`) |

## Use it from other apps

```bat
moex serve --load
```

This starts an API at `http://127.0.0.1:8765/v1` that speaks both OpenAI's and Anthropic's formats, and prints how to
use it. Point any OpenAI-compatible app at it (Open WebUI, editor plugins, agents, the `openai` Python package, ...):
model `qwen3.6-35b-a3b`, any API key. Apps built for Anthropic's API use the base URL `http://127.0.0.1:8765`.

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat, streaming with `"stream": true`, tool calling (`tools` in, `tool_calls` out) |
| `POST /v1/messages` | Anthropic Messages API: system, tools (`tool_use` / `tool_result`), thinking, streaming |
| `POST /v1/messages/count_tokens` | prompt size in tokens (Anthropic format) |
| `POST /v1/completions` | raw text completion without the chat template (editor autocomplete) |
| `GET /v1/models` | the model id |
| `GET /api` | what the server offers, with examples (open it in a browser) |
| `GET /health` | `{"status":"ok"}` |

**Coding agents and tools:** tool calls are parsed from the model's own format and returned in each API's standard
form, with argument types taken from your tool's JSON schema. Prompts are built exactly as Qwen3.6's official chat
template builds them (checked token for token by [`tools/check_template.py`](tools/check_template.py)), and the
official `openai` and `anthropic` Python SDKs run a full call -> result -> answer loop
([`tools/test_tools.py`](tools/test_tools.py)). Agents send long system prompts and tool lists, so give them room:
`moex serve --load --ctx 32768` (each 1,000 tokens of context costs ~10 MB of VRAM).

From PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/v1/chat/completions -Method Post -ContentType application/json -Body '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello!"}]}'
```

From Python (`pip install openai`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="none")
for chunk in client.chat.completions.create(model="qwen3.6-35b-a3b", stream=True,
                                            messages=[{"role": "user", "content": "Hello!"}]):
    print(chunk.choices[0].delta.content or "", end="")
```

Supported: streaming (with `stream_options.include_usage`), `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`max_tokens` (default: until the reply ends or the context is full), `tools`, `tool_choice: "none"`. Not supported:
`n` > 1, `logprobs`, enforced JSON output (`response_format`). Thinking: `moex serve --think`
turns it on for every request; a request can still choose with `chat_template_kwargs.enable_thinking` or
`reasoning_effort` (`"none"` = off). The thinking arrives as `reasoning_content` and the answer as `content`, so Open WebUI
and other apps show it as a separate thinking section. Apps that resend the whole conversation each turn only pay for
the new message.
It listens on your machine only; `--host 0.0.0.0 --port N` changes that (there is no API key, so only on networks you
trust). Web pages can't use it unless they are served from localhost: a site you visit can't send your model
requests or read its replies. To let a web app on another address call moex, start it with `--allow-origin URL`.

## Speed

| engine | decode 1K | decode 8K | prompt reading 1K | prompt reading 8K | VRAM |
|---|---|---|---|---|---|
| **moex** (default) | **67.8** | **64.5** | **733** | **764** | 5.6 GB |
| **moex --fast** | **69.2** | **67.5** | 725 | 765 | 5.6 GB |
| ik_llama.cpp 565f08d1, hand-tuned | 36.0 | 35.1 | 333 | 336 | 5.9 GB |
| llama.cpp b11756, hand-tuned | 33.3 | 32.8 | 202 | 189 | 5.9 GB |

Tokens per second. moex vs the fastest llama engine: decode **1.8-1.9x**, prompt reading **2.2-2.3x**. moex's decode
numbers use its default multi-token prediction (Qwen3.6's own draft head; without it, ~57 tok/s on short prompts);
the llama engines run without speculative decoding.

- Same laptop for all (RTX 4050 Laptop 6 GB, i7-13650HX, 24 GB DDR5), each engine as its own OpenAI-compatible server,
  the same prompts sent as the same token ids (4 of 1,030 tokens and 2 of 8,030), 128 output tokens, greedy, one
  client timing all of them the same way.
- The llama engines run the official ggml-org Q4_K_M GGUF, with the fastest flags found for this card
  (`--n-cpu-moe`, threads, batch sizes, flash attention). moex runs with no flags: it picks threads and cache size itself.
- Exact commands, every flag tried and the RAM numbers: [BENCH_COMPARE.md](BENCH_COMPARE.md). Every measurement is
  logged in [PLAN.md](PLAN.md).
- Short everyday prompts: ~78 tokens/s (`--fast`: 82-85). First token after Enter: under 1 s for short prompts.

**A bigger GPU makes it faster, automatically.** moex turns all free VRAM into expert cache. A 6 GB card holds ~2,050
of the 10,240 experts (20%), an 8 GB card ~2,700 (26%), a 12 GB card about half, and a 24 GB card all of them. Every
expert in VRAM is one the CPU doesn't have to compute. Tested on a laptop that runs its display on the integrated GPU
(the default on most laptops); not yet in the "GPU only" display mode some laptops offer.

**Tested machines** (v0.1.1, `moex chat`; v0.2.0 numbers welcome, see below):

| GPU | CPU | RAM | decode |
|---|---|---|---|
| RTX 4050 Laptop, 6 GB | i7-13650HX | 24 GB | ~46 tokens/s |
| RTX 4060 Laptop, 8 GB | i7-13th gen HX | 24 GB | ~60 tokens/s |

Ran it on something else? Run `moex bench` (3 everyday questions per setting, ~2-4 minutes) and paste the line it prints into a
[benchmark report](https://github.com/neerajdad123-byte/moe/issues/new?template=benchmark.yml).

## How it works

<div align="center">
<img src="docs/how.svg" alt="How moex runs one layer: the router picks 8 of 256 experts; cached ones run on the GPU while missing ones run on the CPU at the same time; missing ones are copied into VRAM in the background" width="760">
</div>

Qwen3.6-35B-A3B is a mixture-of-experts model: 35B parameters in total, but each token uses only 8 of the 256 experts
in each of its 40 layers, about 3B parameters per token. The trick is keeping the right experts close.

- **An expert cache in VRAM.** Experts are cached one at a time, not whole layers, so the 6 GB go to the experts that
  are actually being used right now.
- **Choosing what stays.** Each expert's score is how often it was used, with older uses fading out (half-life of 10
  tokens); the lowest score is evicted. Picked by replaying real routing traces against the alternatives
  ([`tools/sim.py`](tools/sim.py)): it beat LRU, half-lives from 3 to 80 tokens, and predicting the next layer's experts
  (which added nothing). About 70% of the experts each token needs are already in VRAM.
- **Misses on the CPU, in parallel.** For one conversation it is faster to compute a missing expert on the CPU (AVX2,
  and AVX-VNNI int8 where the CPU has it) than to copy it over PCIe. The GPU and CPU work at the same time, and missing
  experts are copied into VRAM in the background for next time.
- **Multi-token prediction.** Qwen3.6's own draft head guesses up to two next tokens; the main model checks them in
  one pass, so most steps produce more than one token (~88% of drafts accepted).
- **Warm start.** moex remembers which experts you use most (`moex_hot.bin` in the model folder; the release ships
  counts from 80 test prompts), and `--load` puts those in VRAM before you type.
- **Everything else on the GPU in int8 / int6.** Attention, the linear-attention (DeltaNet) layers and the output head
  live in VRAM, in int8, with the three largest dense matrices (output head, DeltaNet output, shared-expert down) in
  int6. The KV cache is int8 too (4x smaller), and attention reads each KV head once for its 8 query heads, so long
  conversations stay fast.
- **One launch per token.** The GPU looks up its cached experts itself, results cross PCIe through mapped memory, and a
  whole token (or a whole draft-check round) is a single CUDA graph, with a safe fallback if a graph can't run.
- **Prompts in chunks.** Long prompts are processed 1,024 tokens at a time on tensor cores, streaming each needed expert
  to the GPU once per chunk, ahead of time, while rarely used experts run on the CPU.

## Accuracy

moex runs the published int4 experts as they are, and every release is checked against a PyTorch reference
implementation on three test sets (`moex --check ref/eval1`, `ref/code1`, `ref/agent1`: prose, code and an agent
conversation, 330 positions):

| | mean KL divergence | top-1 token agrees |
|---|---|---|
| default | 0.0024 | 99.4% |
| default, with multi-token prediction | 0.0026 | 99.4% |
| `--fast` | 0.0029-0.0031 | 99.1-99.7% |

KL 0.0024 means the next-token probabilities are nearly identical to the reference's; the top-1 column is how often
both pick the same next token. With full-precision dense weights (`--f16`) the int8 KV cache is the only difference
left: mean KL 0.0005-0.0006. The model file itself is an int4 quantization of Qwen3.6-35B-A3B made by others (see
Credits), so it is not identical to the original BF16 weights.

## Supported today

| | |
|---|---|
| Model | **Qwen3.6-35B-A3B** (int4 gs64 weights from `moex download`). Text only. |
| GPU | NVIDIA RTX 20 / 30 / 40 / 50 series and newer datacenter cards, 6 GB VRAM or more |
| OS | Windows 10/11; Linux (build from source with `./build.sh`; builds on Ubuntu 24.04 under WSL2, speed test pending) |
| RAM | 24 GB recommended. 16 GB works: moex keeps its own RAM cache of experts and reads the rest straight from the SSD (~25-34 tokens/s on the test laptop, +20% since v0.1.3); moex prints a tip to free RAM |
| Use | one conversation at a time, one GPU |

## Roadmap

- **Linux** release binary (today: build from source). Next.
- **GGUF support** and **more models:** load the GGUF files people already have, and other mixture-of-experts
  families (Qwen, DeepSeek, GLM, gpt-oss).
- **More speed:** further lossless experiments, and opt-in modes that trade a little accuracy for speed, measured
  and labelled like `--fast`.
- **A chat page** in the browser, served by `moex serve`.
- Keeping agents' side requests from evicting the main conversation's cache, and several conversations at once in
  `moex serve`.

More plans will be announced soon.

## Building from source

Visual Studio 2026 (C++) and CUDA 13.x. `build.bat` builds `build\moex.exe` for RTX 20-50 series plus the test programs
(`build.bat dev` builds for your GPU only, faster; `build.bat cuda12` builds the older-driver version with CUDA 12.8);
`package.bat` makes the release zips.

Linux: the CUDA toolkit (12.8 or newer) and g++, then `./build.sh` (or `./build.sh dev`) builds `build/moex`. Get the
model with `build/moex download`, then `build/moex chat --load`.

| Check | Command |
|---|---|
| every kernel vs reference outputs | `build\test_kernels.exe` (first: `python ref/qwen36_ref.py "The capital of France is" --n 8 --dump ref/golden`) |
| the whole model vs the reference | `build\moex.exe --check ref/eval1` |
| the tokenizer vs Hugging Face's | `python tools/check_tokenizer.py` |
| the API | `moex serve`, then `python tools/test_server.py` |
| tool calling with the official SDKs | `moex serve`, then `python tools/test_tools.py` (`pip install openai anthropic`) |
| prompts vs Qwen's chat template | `MOEX_DUMP_PROMPT=build\prompt_ids.txt moex serve`, then `python tools/check_template.py` |
| server security and robustness | `moex serve`, then `python tools/test_security.py` and `python tools/test_robust.py` |
| `/reset` in chat | `python tools/test_reset.py` |
| download and PATH helpers (no GPU) | `build\test_download.exe`, `build\test_userenv.exe` |
| the release benchmark | `python tools/bench.py` (1K / 8K prompts, MTP and `--fast`, prompt reading, VRAM, KL) |

The release zips hold only what you need to run moex (`moex.exe`, `moex_hot.bin`, README, LICENSE); every test above
is in this repository. CI builds both zips on every push and runs the checks that need no GPU.

## License and credits

moex is MIT licensed ([LICENSE](LICENSE)). The model has its own license (Apache-2.0).

- Model: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team.
- int4 gs64 weights: [Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64),
  in the format of the [colibri](https://github.com/JustVugg/colibri) project.
- MTP head: converted from the official Qwen weights by [`tools/fetch_mtp.py`](tools/fetch_mtp.py), hosted as the
  [`weights-mtp-1`](https://github.com/neerajdad123-byte/moe/releases/tag/weights-mtp-1) release (Apache-2.0).
- [llama.cpp](https://github.com/ggml-org/llama.cpp), the reference point for speed.
