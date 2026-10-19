"""Tool calling through moex serve with the official SDKs (pip install openai anthropic): the model calls a tool,
gets its result back and answers, streaming and not, over both APIs.

  python tools/test_tools.py [http://127.0.0.1:8765]
"""
import json, sys, time
import anthropic, openai

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
MODEL = "qwen3.6-35b-a3b"
WEATHER = {"Paris": "18C, sunny", "Rome": "24C, cloudy"}
fails = 0

def check(name, ok, detail):
    global fails
    fails += not ok
    print(f"  {name:52s} {'PASS' if ok else 'FAIL'}  {detail}")

# ---- OpenAI ----
oa = openai.OpenAI(base_url=BASE + "/v1", api_key="none")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
for stream in (False, True):
    for think in (False, True):
        msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
        extra = {"chat_template_kwargs": {"enable_thinking": think}}
        t = time.time()
        if stream:
            calls, text, finish = {}, "", None
            for ch in oa.chat.completions.create(model=MODEL, messages=msgs, tools=tools, stream=True, max_tokens=2000, seed=1, extra_body=extra):
                d = ch.choices[0].delta
                text += d.content or ""
                for tc in d.tool_calls or []:
                    c = calls.setdefault(tc.index, {"id": tc.id, "name": "", "args": ""})
                    c["name"] += tc.function.name or ""; c["args"] += tc.function.arguments or ""
                finish = ch.choices[0].finish_reason or finish
            calls = [calls[i] for i in sorted(calls)]
        else:
            r = oa.chat.completions.create(model=MODEL, messages=msgs, tools=tools, max_tokens=2000, seed=1, extra_body=extra)
            m = r.choices[0].message; finish = r.choices[0].finish_reason; text = m.content or ""
            calls = [{"id": c.id, "name": c.function.name, "args": c.function.arguments} for c in m.tool_calls or []]
        args = json.loads(calls[0]["args"]) if calls else {}
        check(f"openai {'stream' if stream else 'plain '} think={think}: calls get_weather",
              finish == "tool_calls" and calls and calls[0]["name"] == "get_weather" and args.get("city") == "Paris",
              f"{time.time() - t:.1f}s finish={finish} calls={[(c['name'], c['args']) for c in calls]} text={text[:60]!r}")
        if not calls: continue
        msgs += [{"role": "assistant", "content": text, "tool_calls": [{"id": c["id"], "type": "function",
                  "function": {"name": c["name"], "arguments": c["args"]}} for c in calls]}]
        msgs += [{"role": "tool", "tool_call_id": c["id"], "content": WEATHER.get(json.loads(c["args"]).get("city"), "unknown")} for c in calls]
        t = time.time()
        r = oa.chat.completions.create(model=MODEL, messages=msgs, tools=tools, max_tokens=2000, seed=1, extra_body=extra)
        ans = r.choices[0].message.content or ""
        check(f"openai {'stream' if stream else 'plain '} think={think}: answers from the result", "18" in ans and r.choices[0].finish_reason == "stop",
              f"{time.time() - t:.1f}s (history resent) {ans[:80]!r}")

# a tool call cut off by max_tokens ("<tool_call>" but no "<function=...>") comes back as text, not dropped
for stream in (False, True):
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        text, ncalls, finish = "", 0, None
        for ch in oa.chat.completions.create(model=MODEL, messages=msgs, tools=tools, stream=True, max_tokens=3, seed=1, extra_body=extra):
            d = ch.choices[0].delta
            text += d.content or ""; ncalls += len(d.tool_calls or []); finish = ch.choices[0].finish_reason or finish
    else:
        r = oa.chat.completions.create(model=MODEL, messages=msgs, tools=tools, max_tokens=3, seed=1, extra_body=extra)
        text, ncalls, finish = r.choices[0].message.content or "", len(r.choices[0].message.tool_calls or []), r.choices[0].finish_reason
    check(f"openai {'stream' if stream else 'plain '}: cut-off tool call kept as text", "<tool_call>" in text and not ncalls and finish == "length",
          f"finish={finish} calls={ncalls} text={text!r}")

# ---- Anthropic ----
an = anthropic.Anthropic(base_url=BASE, api_key="none")
atools = [{"name": "get_weather", "description": "Current weather for a city",
           "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
for stream in (False, True):
    for think in (False, True):
        msgs = [{"role": "user", "content": "What's the weather in Rome? Use the tool."}]
        kw = {"thinking": {"type": "enabled", "budget_tokens": 1024}} if think else {}
        t = time.time()
        if stream:
            with an.messages.stream(model=MODEL, max_tokens=2000, system="You are a helpful assistant.", tools=atools, messages=msgs, **kw) as s:
                for _ in s: pass
                r = s.get_final_message()
        else:
            r = an.messages.create(model=MODEL, max_tokens=2000, system="You are a helpful assistant.", tools=atools, messages=msgs, **kw)
        uses = [b for b in r.content if b.type == "tool_use"]
        thoughts = [b for b in r.content if b.type == "thinking"]
        check(f"anthropic {'stream' if stream else 'plain '} think={think}: tool_use get_weather",
              r.stop_reason == "tool_use" and uses and uses[0].name == "get_weather" and uses[0].input.get("city") == "Rome" and (bool(thoughts) or not think),
              f"{time.time() - t:.1f}s stop={r.stop_reason} blocks={[b.type for b in r.content]} input={uses[0].input if uses else None}")
        if not uses: continue
        msgs += [{"role": "assistant", "content": r.content},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id, "content": WEATHER.get(u.input.get("city"), "unknown")} for u in uses]}]
        t = time.time()
        r = an.messages.create(model=MODEL, max_tokens=2000, system="You are a helpful assistant.", tools=atools, messages=msgs, **kw)
        ans = "".join(b.text for b in r.content if b.type == "text")
        check(f"anthropic {'stream' if stream else 'plain '} think={think}: answers from the result", "24" in ans and r.stop_reason == "end_turn",
              f"{time.time() - t:.1f}s (history resent) {ans[:80]!r}")
n = an.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": "Hello"}], tools=atools)
check("anthropic count_tokens", n.input_tokens > 100, f"{n.input_tokens} tokens")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
