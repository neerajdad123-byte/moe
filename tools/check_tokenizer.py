"""Compare src/tokenizer.h (build/test_tokenizer.exe) with HF tokenizers on a varied corpus.

  python tools/check_tokenizer.py
"""
import glob, os, shutil, subprocess, sys
from tokenizers import Tokenizer

MODEL = os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64")
TJ = os.path.join(MODEL, "tokenizer.json")
OUT = os.path.join("build", "tok_cases")

cases = [
    "Hello, world!", "The capital of France is", "  two spaces", "trailing spaces   ", "tabs\tand\ttabs",
    "line\nbreaks\n\n\nand  \n  mixed \r\n windows\r\n", "I'm sure you're right, they'll see it's done and we'd've",
    "I'M SURE YOU'RE RIGHT, THEY'LL", "don't 'quote' me", "numbers 1234567890 3.14159 -42 1e10 0x1F",
    "émigré café naïve façade Ångström", "中文测试：你好，世界！这是一个句子。", "日本語のテキストです。カタカナ", "한국어 문장입니다",
    "हिन्दी में एक वाक्य", "Русский текст, ёжик", "العربية نص", "emoji 😀🚀👍🏽 family 👨‍👩‍👧 flags 🇮🇳",
    "math ∑∫√∞ ≠ ≤ ≥ ± × ÷ α β γ", "code: for (int i = 0; i < n; ++i) { x[i] += y[i] * 2; }",
    "def f(x):\n    return x ** 2  # square\n\n\nclass A:\n\tpass\n", "SELECT * FROM users WHERE age > 30;",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>",
    "text with <tool_call>{\"a\": 1}</tool_call> inline", "<|im_start|><|im_end|><|im_start|>", "<not a special> token",
    "!!!??? ... --- *** ### @@@", " leading space", "\n\nleading newlines", "a", " ", "\n", "", "   \n\t \n  x",
    "URL https://example.com/path?q=1&r=two#frag and email a.b@c.de", "Ünïcödé MiXeD CaSe wörds",
    "\u00a0nbsp\u00a0separated\u2003em space\u3000ideographic",
]
# long single words (one pretoken each): many merges of equal rank, overlapping pairs, the heap's tie order
import random
rng = random.Random(7)
cases += ["".join(rng.choice("ACGT") for _ in range(20000)), "a" * 5000, "ab" * 3000, "aab" * 2000,
          "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(20000)), "Supercalifragilistic" * 300,
          "".join(rng.choice("éàüößñ") for _ in range(3000)), "ЖжЖж" * 1000]
for pat in ("data/*.txt", "*.md", "src/*.cu", "src/*.h", "tools/*.py", "ref/*.py"):
    for p in sorted(glob.glob(pat)):
        cases.append(open(p, encoding="utf-8", errors="ignore").read())

shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
for i, c in enumerate(cases):
    open(os.path.join(OUT, f"{i:04d}.txt"), "w", encoding="utf-8", newline="").write(c)
r = subprocess.run([os.path.join("build", "test_tokenizer.exe"), TJ, OUT], capture_output=True, text=True)
if r.returncode: print(r.stderr); sys.exit(1)
got = {}
for line in r.stdout.splitlines():
    name, rest = line.split(":", 1)
    got[name] = rest
hf = Tokenizer.from_file(TJ)
bad = 0; total_tok = 0
for i, c in enumerate(cases):
    name = f"{i:04d}.txt"
    want = hf.encode(c, add_special_tokens=False).ids
    rest = got[name]
    rt = "ROUNDTRIP_FAIL" not in rest
    ids = [int(x) for x in rest.replace("ROUNDTRIP_FAIL", "").split()]
    total_tok += len(want)
    if ids != want or not rt:
        bad += 1
        k = next((j for j in range(min(len(ids), len(want))) if ids[j] != want[j]), min(len(ids), len(want)))
        print(f"MISMATCH case {i} ({len(c)} chars){'' if rt else ' [roundtrip]'}: first diff at token {k}: "
              f"got {ids[k:k+6]} want {want[k:k+6]}  text {c[max(0, len(hf.decode(want[:k])) - 10):][:40]!r}")
print(f"{len(cases) - bad}/{len(cases)} cases identical ({total_tok} tokens checked)")
sys.exit(1 if bad else 0)
