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

# thinking: reasoning_content separate from content, streamed and not; a resent history reuses the cached turn
q = [{"role": "user", "content": "Is 91 a prime number? One sentence."}]  # "number only" questions get an empty think block
t = time.time()
r = json.loads(post({"messages": q, "max_tokens": 1500, "seed": 5, "chat_template_kwargs": {"enable_thinking": True}}).read())["choices"][0]["message"]
ok = "<think>" not in r["content"] and "</think>" not in r["content"] and len(r.get("reasoning_content", "")) > 20
print(f"[thinking, non-stream {time.time() - t:.1f}s] reasoning {len(r.get('reasoning_content', ''))} chars | content {r['content']!r} | {'PASS' if ok else 'FAIL'}")
q += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": "And 97? One sentence."}]
t = time.time(); first = None; rs, ct = "", ""
with post({"messages": q, "max_tokens": 1500, "seed": 5, "stream": True, "chat_template_kwargs": {"enable_thinking": True}}) as resp:
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])["choices"][0]["delta"]
        if (d.get("reasoning_content") or d.get("content")) and first is None: first = time.time() - t
        rs += d.get("reasoning_content", ""); ct += d.get("content", "")
ok = "<think>" not in ct and "</think>" not in ct and len(rs) > 20 and ct.strip() != ""
print(f"[thinking, stream, history resent, first chunk {first:.2f}s] reasoning {len(rs)} chars | content {ct!r} | {'PASS' if ok else 'FAIL'}")

# stream_options.include_usage: a last chunk with the counts; GET /v1/models/{id}; no max_tokens: until done
usage = None
with post({"messages": [{"role": "user", "content": "Say hi."}], "stream": True, "stream_options": {"include_usage": True}, "seed": 3}) as resp:
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: {"):
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
print(f"[include_usage] {usage} | {'PASS' if usage and usage['completion_tokens'] > 0 and usage['prompt_tokens'] > 0 else 'FAIL'}")
m = json.loads(urllib.request.urlopen(BASE + "/v1/models/qwen3.6-35b-a3b").read())
print(f"[GET /v1/models/id] {m} | {'PASS' if m.get('context_length') else 'FAIL'}")
r = json.loads(post({"messages": [{"role": "user", "content": "Write two sentences about rain."}], "seed": 3}).read())
print(f"[no max_tokens] {r['usage']['completion_tokens']} tokens, finish {r['choices'][0]['finish_reason']} | {'PASS' if r['choices'][0]['finish_reason'] == 'stop' else 'FAIL'}")

# /v1/completions: raw text continuation, plain and streamed, with a stop string
req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps({"prompt": "def fibonacci(n):\n    ", "max_tokens": 40, "temperature": 0, "stop": ["\n\n"]}).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=600).read())
txt = r["choices"][0]["text"]
print(f"[completions] {txt!r} finish {r['choices'][0]['finish_reason']} | {'PASS' if r['object'] == 'text_completion' and txt.strip() else 'FAIL'}")
req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps({"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, "stream": True}).encode(), headers={"Content-Type": "application/json"})
txt = ""
for line in urllib.request.urlopen(req, timeout=600):
    line = line.decode().strip()
    if line.startswith("data: {"): txt += json.loads(line[6:])["choices"][0]["text"]
print(f"[completions stream] {txt!r} | {'PASS' if 'Paris' in txt else 'FAIL'}")
# the prompt as token ids ("The capital of France is"), and as an array of one string; several prompts: 400
def completions(body):
    req = urllib.request.Request(BASE + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try: return 200, json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")
for prompt in ([760, 6511, 314, 9338, 369], [[760, 6511, 314, 9338, 369]], ["The capital of France is"]):
    st, r = completions({"prompt": prompt, "max_tokens": 8, "temperature": 0, "echo": True})
    txt = r.get("choices", [{}])[0].get("text", "")
    ok = st == 200 and txt.startswith("The capital of France is") and "Paris" in txt[24:] and r["usage"]["prompt_tokens"] == 5
    print(f"[completions prompt {json.dumps(prompt)[:40]}] {st} {txt!r} | {'PASS' if ok else 'FAIL'}")
for prompt in (["one", "two"], [[1, 2], [3, 4]], [760, -1], [760, 1.5], [760, 99999999], [760, "x"]):
    st, r = completions({"prompt": prompt, "max_tokens": 4})
    print(f"[completions prompt {json.dumps(prompt)}] {st} {r.get('error', {}).get('message')!r} | {'PASS' if st == 400 else 'FAIL'}")
