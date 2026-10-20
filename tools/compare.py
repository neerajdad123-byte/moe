# Engine comparison client (BENCH_COMPARE.md): the same token-id prompts to any OpenAI-compatible server
# (moex serve, llama-server, ik_llama.cpp's llama-server) over POST /v1/completions, streamed, greedy, N tokens.
#   python tools/compare.py --url http://127.0.0.1:8080 --prompts build/long1k.txt [--count 16] [--n 128] --pid <server pid>
# One warm-up request first (not counted). Timed on the client, the same way for every engine:
#   prefill tok/s = prompt tokens / time to the first streamed text; decode tok/s = (tokens - 1) / (end - first text).
# Peak VRAM in use (nvidia-smi, whole GPU, minus the --vram-base reading taken before the server started) and the
# server's peak RAM (working set and private bytes) are sampled every 0.5 s.
import argparse, json, subprocess, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='http://127.0.0.1:8080')
ap.add_argument('--prompts', required=True)
ap.add_argument('--count', type=int, default=16)
ap.add_argument('--n', type=int, default=128)
ap.add_argument('--pid', type=int, required=True)
ap.add_argument('--vram-base', type=float, default=0.0, help='MiB in use before the server started')
ap.add_argument('--label', default='')
a = ap.parse_args()

prompts = [[int(x) for x in l.split(',')] for l in open(a.prompts) if l.strip()][:a.count]
peak = {'vram': 0.0, 'ws': 0.0, 'priv': 0.0}
done = False

def sample():
    while not done:
        try:
            v = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).split()[0])
            ws, pv = subprocess.check_output(['powershell', '-NoProfile', '-Command',
                f'$p = Get-Process -Id {a.pid}; "$($p.WorkingSet64) $($p.PrivateMemorySize64)"'], text=True).split()
            peak['vram'] = max(peak['vram'], v); peak['ws'] = max(peak['ws'], int(ws) / 2**30); peak['priv'] = max(peak['priv'], int(pv) / 2**30)
        except Exception:
            pass
        time.sleep(0.5)

def one(ids):
    body = json.dumps({'prompt': ids, 'max_tokens': a.n, 'temperature': 0, 'top_k': 1, 'stream': True, 'ignore_eos': True,
                       'cache_prompt': False, 'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(a.url + '/v1/completions', body, {'Content-Type': 'application/json'})
    t0 = time.perf_counter(); t1 = None; usage = None; chunks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode('utf-8', 'replace').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            d = json.loads(line[5:])
            if d.get('usage'): usage = d['usage']
            if d.get('choices') and d['choices'][0].get('text'):
                chunks += 1
                if t1 is None: t1 = time.perf_counter()
    t2 = time.perf_counter()
    if t1 is None: raise SystemExit(f'no text streamed back (usage {usage})')
    ntok = usage['completion_tokens'] if usage else chunks
    return len(ids) / (t1 - t0), (ntok - 1) / (t2 - t1), ntok

threading.Thread(target=sample, daemon=True).start()
one(prompts[0][-300:])  # warm-up: the prompt's end (the question + assistant turn), so every engine generates
pf, dec, nt = [], [], []
for i, p in enumerate(prompts):
    a1, b1, n1 = one(p)
    pf.append(a1); dec.append(b1); nt.append(n1)
    print(f'  {a.label} prompt {i + 1}/{len(prompts)}: {len(p)} tok, prefill {a1:.0f} tok/s, decode {b1:.2f} tok/s, {n1} tokens', flush=True)
done = True
time.sleep(0.6)
mean = lambda v: sum(v) / len(v)
res = {'label': a.label, 'prompts': len(prompts), 'prompt_tok': round(mean([len(p) for p in prompts])), 'decode': round(mean(dec), 2),
       'decode_min': round(min(dec), 2), 'decode_max': round(max(dec), 2), 'prefill': round(mean(pf), 1), 'gen_tok': f'{min(nt)}-{max(nt)}',
       'vram_mib': round(peak['vram'] - a.vram_base), 'ram_ws_gb': round(peak['ws'], 1), 'ram_private_gb': round(peak['priv'], 1)}
print(json.dumps(res), flush=True)
