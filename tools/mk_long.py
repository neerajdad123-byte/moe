# long-context prompt files: n prompts of ~N tokens each, slices of the repo's text at spread offsets + a question,
# chat template as tok_prompts.py.  usage: mk_long.py N n out.txt
import subprocess, sys
from tokenizers import Tokenizer
N, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
tk = Tokenizer.from_file("C:/models/qwen36_i4_gs64/tokenizer.json")
files = [f for f in subprocess.check_output(['git', 'ls-files'], text=True).split() if f.endswith(('.md', '.cu', '.cuh', '.h', '.py', '.cpp', '.txt')) and 'draft_rank' not in f]
pool = tk.encode('\n\n'.join(open(f, encoding='utf-8', errors='replace').read() for f in files), add_special_tokens=False).ids
q = tk.encode('\n\nSummarize what the text above is about, then list three details from it.', add_special_tokens=False).ids
assert len(pool) > N, (len(pool), N)
stride = (len(pool) - N) // max(n - 1, 1)
lines = [','.join(map(str, [248045, 846, 198] + pool[i * stride:i * stride + N] + q + [248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]))
         for i in range(n)]
open(out, 'w').write('\n'.join(lines) + '\n')
print(len(pool), 'pool tokens,', n, 'prompts of', N, '+', len(q) + 12)
