"""Smoke test of `moex serve` with plain HTTP (what any OpenAI client does).

  python tools/test_server.py [http://127.0.0.1:8765]
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"

def post(body, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)

print(json.loads(urllib.request.urlopen(BASE + "/v1/models").read()))

msgs = [{"role": "user", "content": "My favourite colour is teal. Reply with one short sentence."}]
t = time.time()
r = json.loads(post({"model": "qwen3.6-35b-a3b", "messages": msgs, "max_tokens": 60, "seed": 3}).read())
reply = r["choices"][0]["message"]["content"]
print(f"[non-stream {time.time() - t:.2f}s] {reply!r} | usage {r['usage']} | finish {r['choices'][0]['finish_reason']}")

msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": "What is my favourite colour? One word."}]
t = time.time(); first = None; text = ""
with post({"model": "qwen3.6-35b-a3b", "messages": msgs, "max_tokens": 20, "stream": True, "seed": 3}) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        if d and first is None: first = time.time() - t
        text += d
print(f"[stream, history resent, first chunk {first:.2f}s] {text!r}")

t = time.time()
r = json.loads(post({"messages": [{"role": "user", "content": [{"type": "text", "text": "Count to five."}]}], "max_tokens": 30, "stop": ["4"]}).read())
print(f"[content parts + stop '4', {time.time() - t:.2f}s] {r['choices'][0]['message']['content']!r} finish {r['choices'][0]['finish_reason']}")
