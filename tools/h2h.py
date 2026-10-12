"""Head-to-head: moex v2 vs llama.cpp (and optionally colibri) on the same prompts, template, greedy, N tokens.

  python tools/h2h.py data/h2h_prompts.txt [--n 128] [--only moex|llamacpp|colibri] [--ncmoe 35] [--threads 8]

llama.cpp runs the official ggml-org Q4_K_M GGUF (int4 Q4_K experts, int8 Q8_0 trunk); moex runs the colibri
int4 gs64 container (int4 experts) with its int8 trunk. Same model, same bit widths where it matters.

Both engines run each prompt in a fresh process (cold start, as a user launching a prompt would);
colibri keeps its own persisted warm-start heat file across runs (its feature). moex additionally runs
all prompts in one process (warm cache), reported separately.
decode tok/s = (generated - 1) / (total - TTFT), the same formula for both.
"""
import argparse, json, os, re, subprocess, sys
from tokenizers import Tokenizer

MODEL = os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLI = r"C:\Users\neera\colibri\c"
OUT = os.path.join(ROOT, "build", "h2h")

ap = argparse.ArgumentParser()
ap.add_argument("prompts")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--only")
ap.add_argument("--ncmoe", type=int, default=35)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--with-colibri", action="store_true")
ap.add_argument("--llama-args", default="", help="extra llama.cpp flags, e.g. --no-mmap")
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

tk = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
texts = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" for p in prompts]
res = {"moex": [], "colibri": [], "llamacpp": [], "moex_batch": None}
LLAMA = r"C:\Users\neera\tools\llama.cpp-b11563\bin\llama-completion.exe"
GGUF = r"C:\models\gguf\Qwen3.6-35B-A3B-Q4_K_M.gguf"


def run(cmd, env=None, cwd=None):
    r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout, r.stderr


if a.only in (None, "moex"):
    for i, t in enumerate(texts):
        ids = tk.encode(t).ids
        rc, out, err = run([os.path.join(ROOT, "build", "moex.exe"), "--ids", ",".join(map(str, ids)), "--n", str(a.n)])
        m = re.search(r"\[decode\] (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s \| expert hit ([\d.]+)%", err)
        ttft = re.search(r"TTFT ([\d.]+) s", err)
        gen = [int(x) for x in out.split()]
        row = {"prompt_tokens": len(ids), "generated": len(gen), "decode_tps": float(m.group(3)) if m else None,
               "hit": float(m.group(4)) if m else None, "ttft": float(ttft.group(1)) if ttft else None, "ids": gen, "rc": rc}
        res["moex"].append(row)
        print(f"moex    {i}: {row['decode_tps']} tok/s  TTFT {row['ttft']} s  hit {row['hit']}%  ({len(ids)} prompt tok)", flush=True)
        if rc: print(err[-2000:])
    # warm: all prompts in one process
    ids_file = os.path.join(OUT, "ids.txt")
    open(ids_file, "w").write("".join(",".join(map(str, tk.encode(t).ids)) + "\n" for t in texts))
    rc, out, err = run([os.path.join(ROOT, "build", "moex.exe"), "--batch", ids_file, "--n", str(a.n)])
    rows = re.findall(r"\[prompt (\d+)\] (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s \| expert hit ([\d.]+)%", err)
    res["moex_batch"] = [{"decode_tps": float(r[3]), "hit": float(r[4])} for r in rows]
    tot = re.search(r"\[batch total\] (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s", err)
    print("moex batch (one warm process):", [r["decode_tps"] for r in res["moex_batch"]], "total", tot.group(3) if tot else None, flush=True)

if a.only in (None, "llamacpp"):
    env = dict(os.environ, PATH=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;" + os.environ["PATH"])
    for i, t in enumerate(texts):
        pf = os.path.join(OUT, f"p{i}.txt")
        open(pf, "w", encoding="utf-8", newline="").write(t)
        rc, out, err = run([LLAMA, "-m", GGUF, "-f", pf, "-n", str(a.n), "--temp", "0", "--ignore-eos", "-no-cnv",
                            "--no-display-prompt", "-ngl", "99", "-ncmoe", str(a.ncmoe), "-fa", "on", "-t", str(a.threads),
                            "-c", "4096"] + a.llama_args.split(), env=env)
        pe = re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second", err)
        ev = re.search(r"\beval time =\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second", err)
        row = {"prompt_tokens": int(pe.group(2)) if pe else None, "prefill_tps": float(pe.group(3)) if pe else None,
               "ttft": float(pe.group(1)) / 1e3 if pe else None, "decode_tps": float(ev.group(3)) if ev else None,
               "generated": int(ev.group(2)) + 1 if ev else None, "text": out, "rc": rc}
        res["llamacpp"].append(row)
        print(f"llama.cpp {i}: {row['decode_tps']} tok/s decode  prefill {row['prefill_tps']} tok/s  ({row['prompt_tokens']} prompt tok)", flush=True)
        if rc or not ev: print(err[-3000:])

if a.with_colibri and a.only in (None, "colibri"):
    env = dict(os.environ, SNAP=MODEL, N_NEW=str(a.n), COLI_CUDA="1", COLI_GPUS="0", CUDA_EXPERT_GB="auto",
               Q36_DN_GPU="1", QT_PREFILL_REPLAN="1", HEAT_FILE=os.path.join(OUT, "colibri_heat.bin"),
               OMP_WAIT_POLICY="ACTIVE", OMP_PROC_BIND="close", NOSTREAM="1",
               PATH=os.environ["PATH"] + r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin")
    for i, t in enumerate(texts):
        pf = os.path.join(OUT, f"p{i}.txt")
        open(pf, "w", encoding="utf-8", newline="").write(t)
        rc, out, err = run([os.path.join(COLI, "qwen36.exe"), "256", "4", pf], env=env, cwd=COLI)
        sp = re.search(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)", err)
        ttft = re.search(r"TTFT: ([\d.]+) s", err)
        npt = re.search(r"\[enc\] prompt tokens: (\d+)", err)
        hit = re.search(r"Expert cache hit rate: ([\d.]+)%", err)
        tier = "[gpu] MoE experts -> CUDA VRAM tier" in err
        text = re.search(r"Text\s*: (.*?)\n(?:TTFT|\[|PEAK|$)", err, re.S)
        row = {"prompt_tokens": int(npt.group(1)) if npt else None, "ttft": float(ttft.group(1)) if ttft else None,
               "speed_incl_ttft": float(sp.group(1)) if sp else None, "tier": tier, "rc": rc,
               "text": text.group(1).strip() if text else None}
        if sp and ttft:
            n = int(sp.group(3)); row["decode_tps"] = (n - 1) / (float(sp.group(2)) - float(ttft.group(1)))
        else:
            row["decode_tps"] = None
        row["hit"] = float(hit.group(1)) if hit else None
        res["colibri"].append(row)
        print(f"colibri {i}: {row['decode_tps'] and round(row['decode_tps'], 2)} tok/s  TTFT {row['ttft']} s  "
              f"VRAM tier {'on' if tier else 'OFF'}  ({row['prompt_tokens']} prompt tok)", flush=True)
        if rc or not sp: print(err[-3000:])

json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)

def summary(other, label):
    print(f"\n| # | prompt | moex cold | moex warm | {label} | moex TTFT | {label} TTFT | prompt tok (moex / {label}) |")
    print("|---|---|---|---|---|---|---|---|")
    mw = res["moex_batch"] or [{}] * len(prompts)
    f = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "-"
    for i, p in enumerate(prompts):
        m, c = res["moex"][i], res[other][i]
        print(f"| {i} | {p[:38]} | {f(m['decode_tps'])} | {f(mw[i].get('decode_tps'))} | {f(c['decode_tps'])} | "
              f"{f(m['ttft'])} s | {f(c['ttft'])} s | {m['prompt_tokens']} / {c['prompt_tokens']} |")
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    mc = avg([r["decode_tps"] for r in res["moex"] if r["decode_tps"]])
    mwv = avg([r["decode_tps"] for r in mw if r.get("decode_tps")])
    cc = avg([r["decode_tps"] for r in res[other] if r["decode_tps"]])
    print(f"\nmean decode tok/s: moex cold {mc:.2f} | moex warm {mwv:.2f} | {label} {cc:.2f} | "
          f"moex/{label}: cold {mc / cc:.2f}x, warm {mwv / cc:.2f}x")
    mt = avg([r["ttft"] for r in res["moex"] if r["ttft"]]); ct = avg([r["ttft"] for r in res[other] if r["ttft"]])
    print(f"mean TTFT: moex {mt:.2f} s | {label} {ct:.2f} s")


if res["moex"] and res["llamacpp"]:
    summary("llamacpp", "llama.cpp")

if res["moex"] and res["colibri"]:
    print("\n| # | prompt | moex cold | moex warm | colibri | moex TTFT | colibri TTFT |")
    print("|---|---|---|---|---|---|---|")
    mw = res["moex_batch"] or [{}] * len(prompts)
    for i, p in enumerate(prompts):
        m, c = res["moex"][i], res["colibri"][i]
        f = lambda v: f"{v:.1f}" if isinstance(v, (int, float)) else "-"
        print(f"| {i} | {p[:40]} | {f(m['decode_tps'])} | {f(mw[i].get('decode_tps'))} | {f(c['decode_tps'])} | {f(m['ttft'])} s | {f(c['ttft'])} s |")
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    mc = avg([r["decode_tps"] for r in res["moex"] if r["decode_tps"]])
    mwv = avg([r["decode_tps"] for r in mw if r.get("decode_tps")])
    cc = avg([r["decode_tps"] for r in res["colibri"] if r["decode_tps"]])
    print(f"\nmean decode tok/s: moex cold {mc:.2f} | moex warm {mwv:.2f} | colibri {cc:.2f} | speedup cold {mc / cc:.2f}x, warm {mwv / cc:.2f}x")
    # output agreement: same model, both greedy -> texts should start the same
    agree = 0
    for i in range(len(prompts)):
        mt = tk.decode(res["moex"][i]["ids"], skip_special_tokens=True)
        ct = res["colibri"][i]["text"] or ""
        k = 0
        while k < min(len(mt), len(ct)) and mt[k] == ct[k]: k += 1
        agree += k >= 80
        print(f"  prompt {i}: outputs identical for the first {k} characters")
    print(f"outputs agree on >= 80 chars for {agree}/{len(prompts)} prompts")
