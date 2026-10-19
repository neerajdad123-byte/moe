# A --check set for any prompt (logits from moex itself, e.g. --f16, instead of the PyTorch reference):
#   python tools/make_check_set.py DIR prompt.txt [tokenizer.json]   -> DIR/meta.json (chat-template prompt ids), prints the ids
#   moex --ids <ids> --n 64                                             -> greedy continuation (stdout, one id per line)
#   python tools/make_check_set.py DIR --append cont.txt               -> meta ids = prompt + continuation
#   MOEX_CHECK_DUMP=1 moex --check DIR --f16                            -> DIR/pNNN_logits.bin (the reference)
#   moex --check DIR [--q4 ...]                                         -> KL / top-1 vs that reference
# The prompt file is the user turn; a line "### assistant" / "### user" / "### tool" starts another turn (agent traces).
import json, os, sys

d = sys.argv[1]
os.makedirs(d, exist_ok=True)
if sys.argv[2] == "--append":
    m = json.load(open(os.path.join(d, "meta.json")))
    cont = [int(l) for l in open(sys.argv[3]) if l.strip()]
    m["ids"] = m["ids"][:m["n_prompt"]] + cont
    json.dump(m, open(os.path.join(d, "meta.json"), "w"))
    print(len(m["ids"]), "ids,", m["n_prompt"], "prompt")
    sys.exit(0)
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[3] if len(sys.argv) > 3 else "C:/models/qwen36_i4_gs64/tokenizer.json")
text, role, turns = [], "user", []
for line in open(sys.argv[2], encoding="utf-8").read().split("\n"):
    if line.startswith("### "):
        turns.append((role, "\n".join(text).strip())); role, text = line[4:].strip(), []
    else:
        text.append(line)
turns.append((role, "\n".join(text).strip()))
prompt = ""
for r, t in turns:
    prompt += f"<|im_start|>user\n<tool_response>\n{t}\n</tool_response><|im_end|>\n" if r == "tool" else f"<|im_start|>{r}\n{t}<|im_end|>\n"
prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tk.encode(prompt, add_special_tokens=False).ids
json.dump({"prompt": prompt, "ids": ids, "n_prompt": len(ids)}, open(os.path.join(d, "meta.json"), "w"))
print(",".join(map(str, ids)))
