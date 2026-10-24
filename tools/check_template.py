"""moex's prompts vs the chat template, token for token (tools, tool calls, tool responses, thinking, multi-turn).
Against moex serve: start the server with the prompt dump on, then run this:

  set MOEX_DUMP_PROMPT=build\\prompt_ids.txt & moex serve
  python tools/check_template.py [http://127.0.0.1:8765] [build/prompt_ids.txt]

Offline (no model, no GPU): moex's renderer through build/test_template.exe, e.g. against the template a
`moex import` kept (TEMPLATE_DIR/chat_template.jinja; Qwen's own when the folder has none):

  python tools/check_template.py --offline [TEMPLATE_DIR]

Needs jinja2 and tokenizers (pip install jinja2 tokenizers); without a chat_template.jinja the template comes from
Hugging Face. The tokenizer is QWEN36_MODEL's.
"""
import json, os, subprocess, sys, urllib.request
import jinja2
from tokenizers import Tokenizer

OFFLINE = "--offline" in sys.argv
args = [a for a in sys.argv[1:] if a != "--offline"]
MODEL = os.environ.get("QWEN36_MODEL", r"C:\models\qwen36_i4_gs64")
BASE = args[0] if args and not OFFLINE else "http://127.0.0.1:8765"
DUMP = args[1] if len(args) > 1 and not OFFLINE else os.path.join("build", "prompt_ids.txt")
TPL_DIR = args[0] if args and OFFLINE else MODEL
TPL_FILE = os.path.join(TPL_DIR, "chat_template.jinja")  # an imported GGUF's own template (moex import)
TPL = open(TPL_FILE, encoding="utf-8").read() if os.path.exists(TPL_FILE) else \
    urllib.request.urlopen("https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/chat_template.jinja").read().decode()
print("template:", TPL_FILE if os.path.exists(TPL_FILE) else "Qwen/Qwen3.6-35B-A3B (Hugging Face)")
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))

def raise_exception(m): raise ValueError(m)
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)  # transformers' tojson
env.globals["raise_exception"] = raise_exception
template = env.from_string(TPL)

def reference(messages, tools, think):
    msgs = []
    for m in messages:  # the template wants tool_call arguments as objects (transformers / vLLM parse the string)
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(tc, function=dict(tc["function"], arguments=json.loads(tc["function"]["arguments"])))
                               if isinstance(tc["function"].get("arguments"), str) else tc for tc in m["tool_calls"]]
        msgs.append(m)
    text = template.render(messages=msgs, tools=tools, add_generation_prompt=True, enable_thinking=think)
    return tok.encode(text, add_special_tokens=False).ids

def moex(messages, tools, think):
    if OFFLINE:
        req = os.path.join("build", "template_request.json")
        json.dump({"messages": messages, "tools": tools or [], "think": think}, open(req, "w", encoding="utf-8"), ensure_ascii=False)
        out = subprocess.run([os.path.join("build", "test_template.exe"), os.path.join(MODEL, "tokenizer.json"), TPL_DIR, req],
                             capture_output=True, text=True, check=True).stdout
        return [int(x) for x in out.strip().split(",")]
    body = {"messages": messages, "max_tokens": 1, "chat_template_kwargs": {"enable_thinking": think}}
    if tools: body["tools"] = tools
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=600).read()
    return [int(x) for x in open(DUMP).read().split(",")]

weather = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                           "days": {"type": "integer"}, "units": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["city"]}}}
write = {"type": "function", "function": {"name": "write_file", "description": "Write a file",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                                                         "options": {"type": "object"}}, "required": ["path", "content"]}}}
call = lambda i, name, args: {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

cases = [
    ("system + user", [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi there"}], None, False),
    ("user, thinking", [{"role": "user", "content": "Why is the sky blue?"}], None, True),
    ("multi-turn (older assistant turn)", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
                                           {"role": "user", "content": "Tell me a joke."}], None, False),
    ("tools + system", [{"role": "system", "content": "You help with the weather."}, {"role": "user", "content": "Weather in Paris?"}],
     [weather, write], False),
    ("tool call + 2 tool responses, thinking", [
        {"role": "user", "content": "Weather in Paris and Rome for 3 days?"},
        {"role": "assistant", "content": "I'll check both.", "reasoning_content": "Two cities, call the tool twice.",
         "tool_calls": [call(1, "get_weather", {"city": "Paris", "days": 3}), call(2, "get_weather", {"city": "Rome", "days": 3})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Paris: 18C, sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "Rome: 24C, cloudy"}], [weather], True),
    ("tool call without text, nested + multi-line args", [
        {"role": "user", "content": "Save a note."},
        {"role": "assistant", "content": "", "tool_calls": [call(3, "write_file", {"path": "a/b.txt", "content": "line 1\nline 2", "options": {"mode": "w", "n": 2}})]},
        {"role": "tool", "tool_call_id": "call_3", "content": "ok"},
        {"role": "user", "content": "Thanks. What did you save?"}], [write], False),
    ("multi-turn, earlier turn with reasoning", [{"role": "user", "content": "2+2?"},
                                                 {"role": "assistant", "content": "4", "reasoning_content": "Simple sum."},
                                                 {"role": "user", "content": "And 3+3?"}], None, True),
]
if "merged_system" in TPL:  # finetunes' variant (Unsloth's fixes): developer role, two leading system messages, str() args
    flags = {"type": "function", "function": {"name": "search", "description": "Search",
             "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "exact": {"type": "boolean"},
                                                             "limit": {"type": "integer"}, "tags": {"type": "array"}}}}}
    cases += [
        ("developer + system messages merged", [{"role": "developer", "content": "Be brief."}, {"role": "system", "content": "Answer in French."},
                                                {"role": "user", "content": "Hi"}], None, False),
        ("developer + tools", [{"role": "developer", "content": "Use tools."}, {"role": "user", "content": "Find moex"}], [flags], False),
        ("tool args: bool, null, list", [
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": "", "tool_calls": [call(4, "search", {"q": "moex", "exact": True, "limit": None, "tags": ["a", "b"]})]},
            {"role": "tool", "tool_call_id": "call_4", "content": "3 results"}], [flags], False),
    ]

bad = 0
for name, messages, tools, think in cases:
    r, m = reference(messages, tools, think), moex(messages, tools, think)
    same = r == m
    bad += not same
    msg = f"{len(m)} tokens" if same else f"moex {len(m)} vs template {len(r)} tokens, first difference at {next((i for i, (a, b) in enumerate(zip(r, m)) if a != b), min(len(r), len(m)))}"
    print(f"  {name:45s} {'PASS' if same else 'FAIL'}  ({msg})")
    if not same:
        i = next((i for i, (a, b) in enumerate(zip(r, m)) if a != b), min(len(r), len(m)))
        print("    template:", repr(tok.decode(r[max(0, i - 8):i + 12], skip_special_tokens=False)))
        print("    moex:    ", repr(tok.decode(m[max(0, i - 8):i + 12], skip_special_tokens=False)))
print("ALL PASS" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
