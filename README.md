# moex

Fast local inference for **Qwen3.6-35B-A3B** on a single consumer NVIDIA GPU, even a 6 GB laptop card.

Qwen3.6-35B-A3B is a mixture-of-experts model: 35B parameters in total, but each token uses only 8 of the 256 experts
in every layer. moex keeps the experts that are actually being used in VRAM (expert by expert, not layer by layer),
computes the rest on the CPU while the GPU works, and re-learns what to keep as the conversation moves.

## Speed

Laptop: RTX 4050 6 GB, i7-13650HX, 24 GB DDR5. Same model, same 10 prompts the engines had never seen, same chat
template, greedy decoding, 128 tokens each. llama.cpp b11563 (official CUDA build) on the ggml-org Q4_K_M GGUF, which
has the same bit widths where it matters (int4 experts, int8 everything else).

| | decode, tokens/s | moex is |
|---|---|---|
| **moex** | **41.1** (fresh process) / **42.6** (warm) | |
| llama.cpp, default settings | 20.3 | **2.0x faster** |
| llama.cpp, hand-tuned (`-ngl 99 -ncmoe 33 -t 12 -fa on -lm mmap+mlock`, best of a sweep) | 35.3 | **1.2x faster** |

Time to first token on short chat prompts: ~0.6-0.9 s. Long prompts are processed at ~130 tokens/s.
How these were measured: [`tools/h2h.py`](tools/h2h.py); everything else is in [PLAN.md](PLAN.md).

## Quick start

Requirements: Windows 10/11 · NVIDIA GPU, RTX 20 series or newer, 6 GB VRAM or more · recent NVIDIA driver ·
CPU with AVX2 (any from the last ~10 years) · ~22 GB of disk · 24 GB of RAM recommended
(it runs with less, more slowly). Nothing else: no CUDA toolkit, no Python.

```bat
moex download        :: the model, ~22 GB, resumable (run it again if it stops)
moex chat            :: talk to it
```

In the chat: type a message and press Enter. `/reset` starts a new conversation, `/exit` quits.
Options: `--system "You are ..."`, `--think` (let the model reason before answering), `--temp 0.7 --top-p 0.8 --top-k 20`
(the defaults), `--seed N`, `--ctx 8192`.

### Use it from other apps (OpenAI-compatible API)

```bat
moex serve           :: http://127.0.0.1:8765/v1
```

Point any OpenAI-compatible client (Open WebUI, editors, the `openai` Python package, ...) at
`http://127.0.0.1:8765/v1`, model `qwen3.6-35b-a3b`, any API key. Streaming, `temperature`, `top_p`, `top_k`, `seed`,
`stop`, `max_tokens` and `chat_template_kwargs.enable_thinking` are supported. A client that resends the conversation
each turn only pays for the new message. The server listens on localhost only; use `--host 0.0.0.0 --port N` to
change that.

```bat
curl http://127.0.0.1:8765/v1/chat/completions -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen3.6-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

## How it works

- **Expert cache in VRAM.** About 15% of the 10,240 experts fit next to the rest of the model on a 6 GB card. The ones
  kept are chosen by a decayed use count (half-life 10 tokens), picked by replaying real routing traces against several
  policies (`tools/sim.py`); experts that miss are copied in the background for next time.
- **Misses on the CPU, at the same time.** At batch size 1 computing a missed expert on the CPU is faster than copying
  it over PCIe, so the GPU runs the cached experts while the CPU (AVX2) runs the missing ones.
- **Everything else on the GPU in int8.** Attention, the DeltaNet layers and the output head run from VRAM.
- **No host round trips per layer.** The GPU picks its cached experts itself, results cross PCIe through mapped
  memory, and each layer is one CUDA graph.
- **Prompts in chunks.** Long prompts stream the experts they need to the GPU once per chunk instead of once per token.

## Accuracy

moex runs the published int4 model as it is: with full-precision dense weights it matches a PyTorch reference
implementation to 1e-12 (KL). The default int8 dense weights and int8 KV cache add a mean KL of 0.003 against that
reference, and the top-1 token agrees 98.6% of the time. The model file itself is an int4 quantization of
Qwen3.6-35B-A3B made by others (see Credits), so it is not identical to the original BF16 weights.

## Limitations

- Windows only, one NVIDIA GPU, one conversation at a time.
- This model only (Qwen3.6-35B-A3B, the int4 gs64 container from `moex download`).
- Text only (no images). Context up to ~24k tokens.
- The first reply after start can be slower while the experts load into RAM (~20 s).

## Building from source

Visual Studio 2026 (C++), CUDA 13.x. `build.bat` builds `build\moex.exe` for RTX 20-50 series (`build.bat dev` for
your GPU only, faster) plus the test programs. `build\test_kernels.exe` checks every kernel against reference outputs
(`python ref/qwen36_ref.py ... --dump ref/golden` creates them), `build\moex.exe --check ref/eval1` checks the whole
model, and `python tools/check_tokenizer.py` checks the tokenizer against Hugging Face's.

## Credits

- Model: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) by the Qwen team (Apache-2.0).
- int4 gs64 container: [Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64](https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64)
  (format from the [colibri](https://github.com/JustVugg/colibri) project).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the comparison.
