"""Tokenizing front-end for build/moex.exe (shortcut: Python tokenizer until a C++ one exists).

  python moex.py "Explain MoE routing in one paragraph." --chat --n 128 [--profile]
"""
import argparse, os, subprocess, sys
from tokenizers import Tokenizer

MODEL = os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64")
STOP = "248044,248046"  # <|endoftext|>, <|im_end|>

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--chat", action="store_true", help="wrap in the chat template, thinking off")
ap.add_argument("--profile", action="store_true")
a = ap.parse_args()

tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
text = (f"<|im_start|>user\n{a.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if a.chat else a.prompt)
ids = tk.encode(text).ids
cmd = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "moex.exe"),
       "--ids", ",".join(map(str, ids)), "--n", str(a.n), "--stop", STOP] + (["--profile"] if a.profile else [])
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
out, shown = [], ""
for line in p.stdout:
    out.append(int(line))
    s = tk.decode(out, skip_special_tokens=True)
    sys.stdout.write(s[len(shown):]); sys.stdout.flush()
    shown = s
print()
sys.exit(p.wait())
