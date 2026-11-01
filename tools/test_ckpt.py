"""Prompt checkpoints: a restore after a changed history must give the same greedy tokens as a full re-read.

  python tools/test_ckpt.py [build\\moex.exe] [model dir] [extra serve flags, e.g. --no-mtp]

Starts `moex serve` itself (port 18766, MOEX_DETERMINISTIC=1), then, greedy (temperature 0):
  1. a ~10k-token prompt (this repo's headers; past MOEX_CKPT_EVERY = 8192, so a periodic checkpoint is made too)
     and a short reply
  2. that history with the reply EDITED (the client's copy differs from the tokens moex generated) + a question:
     moex restores the checkpoint at the reply turn's start and reads only the rest
  3. the same request as 2 again (a regenerate): restores the checkpoint before 2's reply turn
  4. twice: an unrelated conversation (resets the engine, drops every checkpoint), then 2's request: a full re-read

The question asks to copy lines from the start of the prompt verbatim. A free-form answer would not do: a full re-read
does not reproduce itself bit for bit (which experts run on the CPU, with int8 activations and other rounding, depends
on the VRAM cache's contents, which move with every request; MTP verify rounds use other kernels than single steps),
so near-ties late in a long answer flip between any two runs, re-read or not. Copying leaves no near-ties, and it only
works if the restored state is right: the K/V rows before the checkpoint and the DeltaNet state at it.
Checks: 2, 3 and both full re-reads give identical text, that text is the copied lines, and 2 and 3 reach their first
token in under half the full re-read's time (the checkpoint was used).
"""
import json, os, subprocess, sys, time, urllib.request

EXE = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "moex" + (".exe" if os.name == "nt" else ""))
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MOEX_MODEL", r"C:\models\qwen36_i4_gs64")
PORT = 18766
BASE = f"http://127.0.0.1:{PORT}"

names = sorted(n for n in os.listdir("src") if n.endswith((".h", ".cuh")))
doc = ""
for name in names:
    doc += f"\n### {name}\n" + open(os.path.join("src", name), encoding="utf-8", errors="ignore").read()
doc = doc[:34000]  # ~10k tokens of code
first = names[0]
want = [l for l in open(os.path.join("src", first), encoding="utf-8", errors="ignore").read().splitlines() if l.strip()][:4]

def ask(messages, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, t_first, text = time.time(), None, ""
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[6:]).get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    t_first = t_first if t_first is not None else time.time() - t0
                    text += piece
    return text, t_first

env = dict(os.environ, MOEX_DETERMINISTIC="1")
proc = subprocess.Popen([EXE, "serve", "--model", MODEL, "--ctx", "32768", "--port", str(PORT)] + sys.argv[3:],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
fails = 0
try:
    for _ in range(600):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=2).read()
            break
        except Exception:
            if proc.poll() is not None:
                sys.exit("moex serve exited")
            time.sleep(1)
    m1 = [{"role": "user", "content": doc + "\n\nIn one sentence: what are these files?"}]
    a1, t1 = ask(m1, 24)
    q = (f"Copy the first {len(want)} non-empty lines of the file {first} above exactly as they appear, "
         "one per line, with nothing before or after them.")
    m2 = m1 + [{"role": "assistant", "content": a1 + " (edited by the client)"}, {"role": "user", "content": q}]
    restored, t_restore = ask(m2, 160)
    regen, t_regen = ask(m2, 160)
    fresh = []
    for _ in range(2):
        ask([{"role": "user", "content": "Say hello."}], 8)  # another conversation: reset, checkpoints dropped
        fresh.append(ask(m2, 160))
    (f1, t_fresh), (f2, _) = fresh
    print(f"first prompt: first token {t1:.2f} s")
    print(f"first token: changed history (checkpoint) {t_restore:.2f} s | regenerate {t_regen:.2f} s | full re-read {t_fresh:.2f} s")
    outs = {"changed history (checkpoint)": restored, "regenerate (checkpoint)": regen, "full re-read 1": f1, "full re-read 2": f2}
    for name, got in outs.items():
        if got != f1:
            fails += 1
            k = next((i for i in range(min(len(got), len(f1))) if got[i] != f1[i]), min(len(got), len(f1)))
            print(f"FAIL {name} differs from full re-read 1 at char {k}:\n  {got!r}\n  {f1!r}")
    copied = [l.strip() for l in f1.strip().strip("`").splitlines() if l.strip() and not l.strip().startswith("```")]
    if copied[:len(want)] != [l.strip() for l in want]:
        fails += 1
        print(f"FAIL the answer is not the copied lines:\n  got  {copied!r}\n  want {[l.strip() for l in want]!r}")
    else:
        print(f"ok   all four answers identical, and they are the first {len(want)} lines of {first}")
    for name, t in (("changed history", t_restore), ("regenerate", t_regen)):
        if not t < 0.5 * t_fresh:
            fails += 1
            print(f"FAIL {name}: {t:.2f} s to the first token, not well under the full re-read's {t_fresh:.2f} s (no checkpoint used?)")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
print("FAILED" if fails else "PASS")
sys.exit(1 if fails else 0)
