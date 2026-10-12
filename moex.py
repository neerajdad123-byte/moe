"""Tokenizing front-end for build/moex.exe (shortcut: Python tokenizer until a C++ one exists).

  python moex.py "Explain MoE routing in one paragraph." --chat --n 128 [--profile] [--f16]
  python moex.py --batch data/prompts.txt --chat --n 128 [--trace build/routes.bin]
"""
import argparse, os, subprocess, sys
from tokenizers import Tokenizer

MODEL = os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64")
STOP = "248044,248046"  # <|endoftext|>, <|im_end|>
EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "moex.exe")

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--chat", action="store_true", help="wrap in the chat template, thinking off")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--f16", action="store_true", help="exact F16 dense weights")
ap.add_argument("--batch", help="file with one prompt per line")
ap.add_argument("--trace", help="write routed-expert trace (with --batch)")
a = ap.parse_args()
if not a.prompt and not a.batch:
    ap.error("give a prompt or --batch")

tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
wrap = lambda p: (f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" if a.chat else p)
flags = ["--n", str(a.n), "--stop", STOP] + (["--profile"] if a.profile else []) + (["--f16"] if a.f16 else [])

if a.batch:
    prompts = [l.strip() for l in open(a.batch, encoding="utf-8") if l.strip()]
    ids_file = os.path.join(os.path.dirname(EXE), "batch_ids.txt")
    with open(ids_file, "w") as f:
        for p in prompts:
            f.write(",".join(map(str, tk.encode(wrap(p)).ids)) + "\n")
    cmd = [EXE, "--batch", ids_file] + flags + (["--trace", a.trace] if a.trace else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for i, line in enumerate(p.stdout):
        out = [int(t) for t in line.split(",") if t.strip()]
        print(f"--- {i}: {prompts[i][:60]}\n{tk.decode(out, skip_special_tokens=True)[:300]}", flush=True)
    sys.exit(p.wait())

ids = tk.encode(wrap(a.prompt)).ids
p = subprocess.Popen([EXE, "--ids", ",".join(map(str, ids))] + flags, stdout=subprocess.PIPE, text=True)
out, shown = [], ""
for line in p.stdout:
    out.append(int(line))
    s = tk.decode(out, skip_special_tokens=True)
    sys.stdout.write(s[len(shown):]); sys.stdout.flush()
    shown = s
print()
sys.exit(p.wait())
