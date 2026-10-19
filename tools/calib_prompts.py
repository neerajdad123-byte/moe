"""Calibration prompts for tools/quant_dense.py: short chat + code prompts as chat-template token ids, one per line
(moex --batch format).

  python tools/calib_prompts.py build/calib_ids.txt [--code 120] [--tokenizer C:/models/qwen36_i4_gs64/tokenizer.json]

Chat: data/prompts.txt (80 everyday / reasoning / code-writing prompts). Code: windows of 10-30 lines from this repo's
sources (C++/CUDA/Python) under rotating instructions (explain, review, comment, refactor, port, test). The check sets
(ref/eval1, code1, agent1) and data/h2h_prompts.txt are left out: they judge the result.
"""
import argparse, glob, os, random

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ASK = [
    "Explain what this code does, step by step.",
    "Review this code: list bugs, edge cases and unclear parts.",
    "Add short comments to this code where they help.",
    "Refactor this code to be simpler without changing what it does.",
    "Port this code to Rust.",
    "Write unit tests for this code.",
    "What is the time and memory cost of this code? Answer briefly.",
    "Rename the variables in this code to clearer names and show the result.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--code", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="C:/models/qwen36_i4_gs64/tokenizer.json")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer)
    rnd = random.Random(a.seed)
    chat = [l.strip() for l in open(os.path.join(ROOT, "data", "prompts.txt"), encoding="utf-8") if l.strip()]
    files = sorted(glob.glob(os.path.join(ROOT, "src", "*.*")) + glob.glob(os.path.join(ROOT, "tools", "*.py")) +
                   glob.glob(os.path.join(ROOT, "ref", "*.py")) + glob.glob(os.path.join(ROOT, "tests", "*.c*")))
    files = [f for f in files if "unicode_tab" not in f]  # a generated table, not code
    srcs = [open(f, encoding="utf-8", errors="replace").read().splitlines() for f in files]
    code = []
    for i in range(a.code):
        lines = rnd.choice([s for s in srcs if len(s) > 60])
        n = rnd.randint(10, 30); s0 = rnd.randint(0, len(lines) - n)
        code.append(ASK[i % len(ASK)] + "\n\n```\n" + "\n".join(lines[s0:s0 + n]) + "\n```")
    with open(a.out, "w") as f:
        for p in chat + code:
            text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            f.write(",".join(map(str, tk.encode(text, add_special_tokens=False).ids)) + "\n")
    print(f"{len(chat)} chat + {len(code)} code prompts -> {a.out}")


if __name__ == "__main__":
    main()
