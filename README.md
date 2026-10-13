# moex

**Run Qwen3.6-35B-A3B at 46 tokens/s on a 6 GB laptop GPU.** About 2.3x faster than llama.cpp as most people run it.

Qwen3.6-35B-A3B is a mixture-of-experts model: 35B parameters, but each token only uses 8 of the 256 experts in each
layer. moex keeps the experts you actually use in VRAM (one expert at a time, not whole layers), runs the rest on the
CPU at the same time, and keeps re-learning what to keep as the conversation moves.

## Speed

Laptop: RTX 4050 6 GB, i7-13650HX, 24 GB DDR5. Same model, 10 prompts neither engine had seen, same chat template,
greedy decoding, 128 tokens each. llama.cpp b11563 (official CUDA build) on the ggml-org Q4_K_M GGUF, which has the same
bit widths where it matters (int4 experts, int8 everything else).

| | decode, tokens/s |
|---|---|
| **moex** | **46.2** average (41-53 depending on the prompt) |
| llama.cpp, default settings | 20.3 |

- First token on a chat message: ~0.4-0.9 s. Long prompts are read at ~110-140 tokens/s.
- Long context: a 26,500-token prompt still decodes at ~30 tokens/s.
- How it was measured: [`tools/h2h.py`](tools/h2h.py). Every measurement is logged in [PLAN.md](PLAN.md).

## Supported models and GPUs

| | |
|---|---|
| Models | **Qwen3.6-35B-A3B**, int4 (gs64) weights, downloaded by `moex download`. More are on the roadmap. |
| GPUs | NVIDIA RTX 20 / 30 / 40 / 50 series and newer datacenter cards, 6 GB VRAM or more |
| OS | Windows 10/11 (Linux next) |

**More VRAM is used automatically.** At startup moex turns all free VRAM (minus a small margin for the driver) into
expert cache. On a 6 GB card that's ~1,600 experts (15% of them); an 8 GB card holds ~1,100 more (~26%), 12 GB about half,
and a 24 GB card holds every expert. More experts in VRAM means fewer run on the CPU, so it gets faster with no setting
to change.

## Quick start (Windows)

You need: Windows 10/11 · an NVIDIA GPU, RTX 20 series or newer, 6 GB VRAM or more · a recent NVIDIA driver · ~22 GB of
free disk · 24 GB of RAM recommended (less works, more slowly). Nothing else: no CUDA toolkit, no Python.

1. Download `moex-windows-x64.zip` from [Releases](../../releases) and unzip it anywhere.
2. Open a terminal in that folder and run:

```bat
moex download          :: the model, ~22 GB, resumable (run it again if it stops)
moex chat --load       :: load everything up front, then chat at full speed
```

`--load` takes ~30 s and then every reply runs at full speed. Without it the prompt is ready in ~7 s, but the first
replies are slower while the model finishes loading into RAM in the background.

At startup moex prints free RAM and VRAM before and after loading. Afterwards both look almost full: that's on purpose,
free VRAM becomes expert cache and the experts are held in RAM. After each reply it prints the speed:

```
[after load] RAM 0.3 GB free of 25.5 GB | VRAM 0.3 GB free of 6.4 GB  (by design: ...)

Now the stage is yours.

> Explain in 3 sentences why the sky is blue.
Sunlight reaches Earth's atmosphere and is scattered ...
[60 tokens, 44.5 tok/s, first token 0.43 s, context 84]
```

In the chat: type and press Enter. `/reset` starts a new conversation, `/exit` quits.

| Option | |
|---|---|
| `--load` | preload the most-used experts into VRAM and all experts into RAM before the prompt |
| `--system "You are ..."` | system prompt |
| `--think` | let the model reason before answering |
| `--temp 0.7 --top-p 0.8 --top-k 20` | sampling (these are the defaults; `--temp 0` = greedy) |
| `--seed N` | reproducible sampling |
| `--ctx 8192` | context length in tokens (each extra 1,000 tokens costs ~10 MB of VRAM) |
| `--model DIR` | model folder (default: `models\` next to `moex.exe`) |

## Use it from other apps (OpenAI-compatible API)

```bat
moex serve --load      :: http://127.0.0.1:8765/v1
```

Point any OpenAI-compatible client (Open WebUI, editor plugins, the `openai` Python package, ...) at
`http://127.0.0.1:8765/v1`, model `qwen3.6-35b-a3b`, any API key. Supported: streaming, `temperature`, `top_p`,
`top_k`, `seed`, `stop`, `max_tokens`, `chat_template_kwargs.enable_thinking`. A client that resends the whole
conversation each turn only pays for the new message. It listens on localhost only; `--host 0.0.0.0 --port N` changes
that.

```bat
curl http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen3.6-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

## How it works

- **Expert cache in VRAM.** About 15% of the 10,240 experts fit next to the rest of the model on a 6 GB card.
- **Which experts stay: a decayed use count.** Each expert's score is how often it was used, with older uses fading
  out (half-life: 10 tokens). The lowest score is evicted. This was chosen by replaying real routing traces against
  the other policies (`tools/sim.py`): it beat LRU, half-lives from 3 to 80 tokens, and next-layer (Markov)
  prefetching, which added nothing on real traces. It gets 73% of experts from VRAM; a perfect oracle that knows the
  future would get 86%.
- **Misses on the CPU, at the same time.** At batch size 1, computing a missing expert on the CPU (AVX2) is faster
  than copying it over PCIe. The GPU runs the cached experts while the CPU runs the missing ones, and the missing ones
  are copied into VRAM in the background for next time.
- **Warm start.** moex counts how often each expert is used across your sessions (`moex_hot.bin` in the model folder;
  the release ships counts from 80 test prompts). `--load` fills VRAM with the most-used ones before you type.
- **Everything else on the GPU in int8.** Attention, the DeltaNet layers and the output head run from VRAM. The KV
  cache is int8 too (4x smaller), and attention is split across the GPU (flash-decoding), so long contexts stay fast.
- **No host round trips per layer.** The GPU looks up its cached experts itself, results cross PCIe through mapped
  memory, and each layer is one CUDA graph.
- **Prompts in chunks.** Long prompts stream the experts they need to the GPU once per chunk of 256 tokens instead of
  once per token.

## Accuracy

moex runs the published int4 model as it is. With full-precision dense weights (`--f16`) it matches a PyTorch
reference implementation to 1e-12 (KL). The default int8 dense weights and int8 KV cache add a mean KL of 0.003, and
the top-1 token agrees with the reference 98.6% of the time. The model file is itself an int4 quantization of
Qwen3.6-35B-A3B made by others (see Credits), so it is not identical to the original BF16 weights.

## Limitations

- Windows only for now (Linux is next). One NVIDIA GPU, one conversation at a time.
- This model only: Qwen3.6-35B-A3B, the int4 gs64 files that `moex download` fetches. Text only, no images.
- Tested on one laptop so far (RTX 4050 6 GB, 24 GB RAM). Reports from other machines are welcome: open an issue
  with the `[load]` lines and the tok/s you see.

## Roadmap

Next up, roughly in order:

- **Linux** build.
- **Aggressive mode:** fit more experts on the GPU (int4 for the dense layers that tolerate it, smaller buffers,
  context sized to what you use), and less RAM needed, so 16 GB machines run well.
- **More models:** other mixture-of-experts models (Qwen, DeepSeek, GLM, gpt-oss families).
- **GGUF support:** load the GGUF files people already have, natively, without converting.
- **MTP (multi-token prediction):** Qwen3.6 can draft extra tokens per step; the int4 files we use today don't include
  that head, so this comes with the next model format.
- Faster prompt reading and multiple conversations at once for `moex serve`.

More plans will be announced soon.

## Building from source

Visual Studio 2026 (C++), CUDA 13.x. `build.bat` builds `build\moex.exe` for RTX 20-50 series (`build.bat dev`:
your GPU only, faster) plus the test programs; `package.bat` makes the release zip.

| Check | Command |
|---|---|
| every kernel vs reference outputs | `build\test_kernels.exe` (needs `python ref/qwen36_ref.py "The capital of France is" --n 8 --dump ref/golden`) |
| the whole model vs the reference | `build\moex.exe --check ref/eval1` |
| the tokenizer vs Hugging Face's | `python tools/check_tokenizer.py` |
| the API | `moex serve`, then `python tools/test_server.py` |

## License

MIT (see [LICENSE](LICENSE)) for the moex code. The model has its own license (Apache-2.0).

## Credits

- Model: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team (Apache-2.0).
- int4 gs64 files: [Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64)
  (format from the [colibri](https://github.com/JustVugg/colibri) project).
- [llama.cpp](https://github.com/ggml-org/llama.cpp), the comparison.
