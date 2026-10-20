# One-command speed + quality benchmark of a build (the release gates), from the repo root after build.bat:
#   python tools/bench.py [--exe build/moex.exe] [--model C:/models/qwen36_i4_gs64] [--ctx 1k,8k] [--prompts 16]
#                         [--tokens 64] [--rounds 2] [--no-check]
# Prompts: slices of this repo's text at spread offsets (~N tokens each) + a question, chat template. A warm-up run
# (expert file cache), then per context: rounds x (raw, MTP) x (default, --fast), alternating; decode tok/s (batch
# total), prefill tok/s, VRAM hit rate, MTP acceptance. Then --check ref/eval1: mean KL / top-1 vs the f16 reference.
# Prints markdown tables (paste into PLAN.md). Needs `pip install tokenizers`. `moex bench` is the user-facing one.
import argparse, re, subprocess, sys, tempfile, os
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--exe', default='build/moex.exe')
ap.add_argument('--model', default='C:/models/qwen36_i4_gs64')
ap.add_argument('--ctx', default='1k,8k', help='prompt sizes, e.g. 1k,8k,32k')
ap.add_argument('--prompts', type=int, default=16)
ap.add_argument('--tokens', type=int, default=64)
ap.add_argument('--rounds', type=int, default=2)
ap.add_argument('--no-check', action='store_true')
a = ap.parse_args()
a.exe = os.path.abspath(a.exe)  # Windows does not search relative paths with /

tk = Tokenizer.from_file(os.path.join(a.model, 'tokenizer.json'))
files = [f for f in subprocess.check_output(['git', 'ls-files'], text=True).split()
         if f.endswith(('.md', '.cu', '.cuh', '.h', '.py', '.cpp', '.txt')) and 'draft_rank' not in f]
pool = tk.encode('\n\n'.join(open(f, encoding='utf-8', errors='replace').read() for f in files), add_special_tokens=False).ids
q = tk.encode('\n\nSummarize what the text above is about, then list three details from it.', add_special_tokens=False).ids
tmp = tempfile.mkdtemp(prefix='moex_bench_')

def prompt_file(n_tok):
    assert len(pool) > n_tok, f'the repo text has {len(pool)} tokens, < {n_tok}'
    stride = (len(pool) - n_tok) // max(a.prompts - 1, 1)
    path = os.path.join(tmp, f'p{n_tok}.txt')
    with open(path, 'w') as f:
        for i in range(a.prompts):
            ids = [248045, 846, 198] + pool[i * stride:i * stride + n_tok] + q + [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
            f.write(','.join(map(str, ids)) + '\n')
    return path

def run(args):
    r = subprocess.run([a.exe, '--model', a.model] + args, capture_output=True, text=True, errors='replace')
    if r.returncode:
        print(f'  ! exit {r.returncode}: {" ".join(args)}\n' + r.stderr[-1500:], file=sys.stderr)
    return r.stderr

def num(pat, s, default=None):
    m = re.findall(pat, s, re.M)
    return float(m[-1]) if m else default

ctxs = [int(c.lower().rstrip('k')) * 1000 for c in a.ctx.split(',')]
print('warm-up ...', file=sys.stderr)
run(['--batch', prompt_file(1000), '--n', '16', '--ctx', '4096', '--load', '--no-mtp'])
rows = {}
for n_tok in ctxs:
    path, ctx = prompt_file(n_tok), str(n_tok + 1024 + a.tokens)
    for r in range(a.rounds):
        for mode in ('no-mtp', 'mtp'):
            for fast in ('', '--fast'):
                print(f'{n_tok // 1000}K round {r + 1}: {mode} {fast or "default"} ...', file=sys.stderr)
                e = run(['--batch', path, '--n', str(a.tokens), '--ctx', ctx, '--load', '--' + mode] + ([fast] if fast else []))
                pf = re.findall(r'^\[prefill\] (\d+) tok in ([\d.]+) s', e, re.M)
                d = rows.setdefault((n_tok, mode, fast), {'dec': [], 'pf': [], 'hit': [], 'acc': []})
                d['dec'].append(num(r'^\[batch total\] .*?= ([\d.]+) tok/s', e, 0.0))
                d['pf'].append(sum(int(t) for t, _ in pf) / max(sum(float(s) for _, s in pf), 1e-9))
                d['hit'].append(num(r'^\[batch total\] .*?hit ([\d.]+)%', e, 0.0))
                if mode == 'mtp': d['acc'].append(num(r'accepted ([\d.]+)%', e, 0.0))
                d['vram'] = (num(r'KV cache (\d+)', e), num(r'expert cache (\d+) slots', e))

f1 = lambda v: ' / '.join(f'{x:.1f}' for x in v)
print(f'\n{a.prompts} prompts x {a.tokens} tokens, warm-up, {a.rounds} alternating rounds ({a.exe})\n')
print('| context | mode | decode tok/s | prefill tok/s | VRAM hit | acceptance | KV MB / expert slots |')
print('|---|---|---|---|---|---|---|')
for (n_tok, mode, fast), d in rows.items():
    m = ('MTP' if mode == 'mtp' else 'raw') + (' --fast' if fast else '')
    print(f'| {n_tok // 1000}K | {m} | **{f1(d["dec"])}** | {sum(d["pf"]) / len(d["pf"]):.0f} | {f1(d["hit"])}% | '
          f'{f1(d["acc"]) + "%" if d["acc"] else "-"} | {d["vram"][0]:.0f} / {d["vram"][1]:.0f} |')

if not a.no_check:
    print('\n| quality (ref/eval1) | mean KL | top-1 |\n|---|---|---|')
    for mode in ('--no-mtp', '--mtp'):
        for fast in ('', '--fast'):
            e = run(['--check', 'ref/eval1', '--load', mode] + ([fast] if fast else []))
            n, mm = num(r'(\d+) positions', e, 1), num(r'\((\d+) mismatch', e, 0)
            print(f'| {"MTP" if mode == "--mtp" else "raw"}{" --fast" if fast else ""} | {num(r"mean KL ([\d.e+-]+)", e, float("nan")):.2e} | {100 - 100 * mm / n:.1f}% |')
