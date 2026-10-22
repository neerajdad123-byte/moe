"""llama.cpp on a GGUF vs moex's --check reference outputs (ref/eval1 etc.): mean KL(ref || llama) and top-1 agreement
over the same scored positions as `moex --check` (the prompt's last position to the end, teacher-forced).

llama-server gives no full logits, only the top-N next-token probabilities (from the raw logits, before sampling), so
KL uses llama's top N and spreads the rest of its mass evenly over the other tokens. The reference's probability mass
outside llama's top N is printed: when it is ~0 the KL is exact to that size.

  python tools/llama_check.py GGUF SET... [--server PATH] [--n-probs 1000]
"""
import argparse, json, math, os, subprocess, sys, time, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("gguf")
ap.add_argument("sets", nargs="+")
ap.add_argument("--server", default=r"C:\Users\neera\tools\llama.cpp-b11756\bin\llama-server.exe")
ap.add_argument("--n-probs", type=int, default=1000)
ap.add_argument("--port", type=int, default=8093)
a = ap.parse_args()
V = 248320
URL = f"http://127.0.0.1:{a.port}"

# the BENCH_COMPARE.md settings for this laptop (6 GB VRAM)
srv = subprocess.Popen([a.server, "-m", a.gguf, "-ngl", "99", "-fa", "on", "--n-cpu-moe", "32", "-t", "14", "-c", "4096",
                        "-np", "1", "--port", str(a.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(600):
        try:
            if json.loads(urllib.request.urlopen(URL + "/health", timeout=2).read()).get("status") == "ok": break
        except Exception: pass
        if srv.poll() is not None: sys.exit("llama-server exited")
        time.sleep(1)

    def top(ids):
        body = {"prompt": ids, "n_predict": 1, "n_probs": a.n_probs, "temperature": 0, "cache_prompt": True}
        r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/completion", data=json.dumps(body).encode(),
                                                                     headers={"Content-Type": "application/json"}), timeout=600).read())
        c = r["completion_probabilities"][0]
        if "top_logprobs" in c: return {t["id"]: math.exp(t["logprob"]) for t in c["top_logprobs"]}
        return {t["id"]: t["prob"] for t in c["probs"]}

    for s in a.sets:
        m = json.load(open(os.path.join(s, "meta.json")))
        ids, np_ = m["ids"], m["n_prompt"]
        kls, agree, n, tail = [], 0, 0, []
        for pos in range(np_ - 1, len(ids)):
            f = os.path.join(s, f"p{pos:03d}_logits.bin")
            if not os.path.exists(f): continue
            ref = np.fromfile(f, dtype=np.float32).astype(np.float64)
            lp = ref - ref.max(); lp -= np.log(np.exp(lp).sum()); pr = np.exp(lp)
            q = top(ids[:pos + 1])
            rest = max(1.0 - sum(q.values()), 1e-30) / (V - len(q))
            lq = np.full(V, math.log(rest))
            for i, p in q.items(): lq[i] = math.log(max(p, 1e-30))
            kls.append(float((pr * (lp - lq)).sum()))
            agree += max(q, key=q.get) == int(ref.argmax())
            tail.append(float(1.0 - pr[list(q)].sum()))
            n += 1
        print(f"[llama] {s}: {n} positions, mean KL {np.mean(kls):.2e}, worst KL {np.max(kls):.2e}, top-1 agreement "
              f"{100 * agree / n:.1f}% ({n - agree} mismatch), ref mass outside llama's top {a.n_probs}: max {max(tail):.1e}", flush=True)
finally:
    srv.kill()
