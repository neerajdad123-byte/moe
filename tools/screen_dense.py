"""Screen dense-weight formats end to end: `moex --check` on several sets per config, position-weighted mean KL and
top-1 over all of them, against the first config (the int8 baseline). PLAN.md's budget for a whole mix: KL +1e-3,
top-1 -0.5 points; run-to-run noise ~+-0.15e-3.

Grid (the per-tensor table; quantizes missing configs with tools/quant_dense.py first, then one row per config):
  python tools/screen_dense.py --grid --sets ref/eval1,../moe-speed/build/ref_code1,../moe-speed/build/ref_agent1 \
      --groups attn_qkv,attn_o,sh_gu,sh_d,dn_in,dn_out,lm_head --bits 4,6 --methods fq,awq,gptq,awq+gptq \
      --calib build/calib --qroot build/q [--gs 64] [--asym] [--qargs "--act-order"]
  method fq = the engine's own RTN through MOEX_FQ=group:bits:gs:asym (no file); the others load MOEX_DENSE=qroot/...
Free form: python tools/screen_dense.py --sets ... base= gptq4=MOEX_DENSE=build/q/x fq6=MOEX_FQ=lm_head:6:64:0 q6=--q6:lm_head
  (a config is NAME=SETTINGS, SETTINGS a '+'-joined list of ENV=VALUE or --flag:value; empty = int8 baseline)
Output: one line per config as it finishes, then a markdown table (also written to --table-out).
"""
import argparse, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def gpu_busy():
    t = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    return any(l.startswith(("moex", "test_kernels")) for l in t.splitlines())  # moex_a.exe etc. too


class Lock:  # --lock PATH --lock-name NAME: the shared GPU lock, taken for each moex run only (others interleave)
    def __init__(self, a): self.path, self.name = a.lock, a.lock_name
    def __enter__(self):
        while self.path:
            while os.path.exists(self.path) or gpu_busy(): time.sleep(5)
            try:  # O_EXCL: never overwrite a lock someone else just took
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, self.name.encode()); os.close(fd)
                break
            except FileExistsError:
                continue
        return self
    def __exit__(self, *e):
        try:
            if self.path and open(self.path).read().strip() == self.name: os.remove(self.path)
        except OSError: pass


def run_check(a, s, settings):
    env, args = dict(os.environ), []
    for kv in filter(None, settings.split("+")):
        if kv.startswith("--"):
            f, _, v = kv.partition(":"); args += [f] + ([v] if v else [])
        else:
            k, _, v = kv.partition("="); env[k] = v
    with Lock(a):
        r = subprocess.run([a.exe, "--check", s, "--model", a.model] + args + a.extra.split(), env=env, capture_output=True, text=True)
    line = next((l for l in reversed(r.stderr.splitlines()) if l.startswith("[check]")), None)
    if not line:
        sys.exit(f"no [check] line for {settings!r} on {s}:\n{r.stderr[-2000:]}")
    n = int(re.search(r"(\d+) positions", line).group(1))
    kl = float(re.search(r"mean KL ([0-9.e+-]+)", line).group(1))
    mm = int(re.search(r"\((\d+) mismatch", line).group(1))
    return n, kl, mm


def grid_configs(a):
    cfgs = [("int8", "", "-", "-", "-")]
    for g in a.groups.split(","):
        for b in a.bits.split(","):
            for m in a.methods.split(","):
                if m == "fq":
                    cfgs.append((f"fq {g} {b}", f"MOEX_FQ={g}:{b}:{a.gs}:{int(a.asym)}", g, b, "rtn (MOEX_FQ)"))
                    continue
                d = os.path.join(a.qroot, f"{m}{b}_g{a.gs}{'a' if a.asym else ''}_{g}")
                if not os.path.exists(os.path.join(d, "dense.safetensors")):
                    cmd = [sys.executable, os.path.join(HERE, "quant_dense.py"), "--calib", a.calib, "--out", d, "--groups", g,
                           "--bits", b, "--gs", str(a.gs), "--method", m] + (["--asym"] if a.asym else []) + a.qargs.split()
                    print("quantizing:", " ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)
                cfgs.append((f"{m} {g} {b}", f"MOEX_DENSE={d}", g, b, m + (" " + a.qargs if a.qargs else "")))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--sets", default="ref/eval1")
    ap.add_argument("--exe", default="build/moex.exe")
    ap.add_argument("--model", default=os.environ.get("QWEN36_MODEL", "C:/models/qwen36_i4_gs64"))
    ap.add_argument("--extra", default="", help="more moex args for every run")
    ap.add_argument("--kl-budget", type=float, default=1e-3); ap.add_argument("--top1-budget", type=float, default=0.5)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--groups", default="attn_qkv,attn_o,sh_gu,sh_d,dn_in,dn_out,lm_head")
    ap.add_argument("--bits", default="4,6"); ap.add_argument("--methods", default="fq,awq,gptq,awq+gptq")
    ap.add_argument("--gs", type=int, default=64); ap.add_argument("--asym", action="store_true")
    ap.add_argument("--calib", default="build/calib"); ap.add_argument("--qroot", default="build/q")
    ap.add_argument("--qargs", default="", help="more quant_dense.py args, e.g. '--act-order --clip'")
    ap.add_argument("--table-out", default="")
    ap.add_argument("--lock", default="", help="shared GPU lock file, held only while moex runs")
    ap.add_argument("--lock-name", default="agent2")
    a = ap.parse_args()
    cfgs = grid_configs(a) if a.grid else [(c.partition("=")[0], c.partition("=")[2], "-", "-", "-") for c in a.configs]
    sets = a.sets.split(",")
    rows, base = [], None
    for name, settings, g, b, m in cfgs:
        per, tn, tk, tm = [], 0, 0.0, 0
        for s in sets:
            n, kl, mm = run_check(a, s, settings)
            per.append((kl, mm)); tn += n; tk += kl * n; tm += mm
        kl, top1 = tk / tn, 100 - 100 * tm / tn
        if base is None:
            base, dk, dt, verdict = (kl, top1), 0.0, 0.0, "baseline"
        else:
            dk, dt = kl - base[0], top1 - base[1]
            verdict = "within budget" if dk <= a.kl_budget and dt >= -a.top1_budget else "over budget"
        rows.append((g, b, m, per, kl, top1, dk, dt, verdict))
        print(f"[{name}] " + " | ".join(f"{os.path.basename(s)} {k:.2e}/{x}" for s, (k, x) in zip(sets, per)) +
              f" | all: mean KL {kl:.2e}, top-1 {top1:.1f}% | KL added {dk:+.2e}, top-1 {dt:+.1f} | {verdict}", flush=True)
    head = "| tensor | bits | method | " + " | ".join(f"{os.path.basename(s)} KL / miss" for s in sets) + \
           " | mean KL | KL added | top-1 | top-1 change | vs budget |"
    out = [head, "|" + "---|" * (len(sets) + 8)]
    for g, b, m, per, kl, top1, dk, dt, v in rows:
        out.append(f"| {g} | {b} | {m} | " + " | ".join(f"{k:.2e} / {x}" for k, x in per) +
                   f" | {kl:.2e} | {dk:+.2e} | {top1:.1f}% | {dt:+.1f} | {v} |")
    print("\n".join(out))
    if a.table_out: open(a.table_out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
