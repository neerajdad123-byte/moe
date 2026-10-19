"""`/reset` in moex chat must start from a clean model state: after a first conversation and /reset, the reply to a
question must be exactly what a fresh process replies (greedy decoding).

  python tools/test_reset.py [path\\to\\moex.exe] [model dir]
"""
import os, subprocess, sys

EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "moex.exe")
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MOEX_MODEL", r"C:\models\qwen36_i4_gs64")
Q = "What is the secret word? If you were never told one, answer exactly: UNKNOWN."

def chat(lines):
    r = subprocess.run([EXE, "chat", "--model", MODEL, "--temp", "0", "--n", "40", "--no-mtp"], input="\n".join(lines + ["/exit"]) + "\n",
                       capture_output=True, text=True, encoding="utf-8", timeout=900)
    return [x.strip() for x in r.stdout.split("> ") if x.strip()]  # one entry per reply

OPEN = "Write a few sentences about the sea."  # open-ended: 40 greedy tokens show any leftover state
FIRST = ["Remember this: the secret word is PINEAPPLE. I love pineapples, tropical islands and summer. Reply with OK."]
fresh = chat([Q])[0], chat([OPEN])[0]
after = chat(FIRST + ["/reset", Q])[-1], chat(FIRST + ["/reset", OPEN])[-1]
ok = True
for f, a in zip(fresh, after):
    print(f"fresh process:   {f!r}\nafter /reset:    {a!r}\n")
    ok &= f == a and "PINEAPPLE" not in a.upper()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
