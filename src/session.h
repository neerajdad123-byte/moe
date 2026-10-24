// A conversation on top of the Engine: the engine state (KV cache, DeltaNet state) is kept between turns, so a new
// turn only prefills its new tokens. Prompts follow Qwen3.6's chat template (tools, tool calls, tool responses,
// thinking). Included by moex.cu after Engine.
#pragma once
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "json.h"
#include "tokenizer.h"

struct ToolCall { std::string id, name; Json args; };  // args: an object
struct ChatMsg {
    std::string role, content;
    std::string reasoning; bool has_reasoning = false;  // assistant: reasoning_content sent back by the client
    std::vector<ToolCall> tool_calls;                   // assistant
};

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n\f\v"), b = s.find_last_not_of(" \t\r\n\f\v");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}
static bool starts_with(const std::string& s, const std::string& p) { return !s.compare(0, p.size(), p); }
static bool ends_with(const std::string& s, const std::string& p) { return s.size() >= p.size() && !s.compare(s.size() - p.size(), p.size(), p); }

// The template's tool instructions (Qwen3.6 chat_template.jinja), around one tojson line per tool.
static const std::string TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const std::string TOOLS_RULES =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must "
    "be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional "
    "reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function "
    "call available, answer the question like normal with your current knowledge and do not tell the user about function "
    "calls\n</IMPORTANT>";

// A tool's JSON-schema type for one parameter ("" if unknown). tools: OpenAI form {"type":"function","function":{...}}.
static std::string param_type(const std::vector<Json>& tools, const std::string& fn, const std::string& param) {
    for (auto& t : tools) {
        const Json* f = t.get("function"); if (!f) f = &t;
        if (f->str("name") != fn) continue;
        const Json* pr = f->get("parameters"); const Json* props = pr ? pr->get("properties") : nullptr;
        const Json* p = props ? props->get(param) : nullptr;
        const Json* ty = p ? p->get("type") : nullptr;
        if (ty && ty->t == Json::STR) return ty->s;
        if (ty && ty->t == Json::ARR) for (auto& x : ty->a) if (x.t == Json::STR && x.s != "null") return x.s;
        return p ? "object" : "";  // a schema without "type" (anyOf, ...): parse the value as JSON if it is
    }
    return "";
}

static std::string new_call_id(const char* prefix) {
    static std::mt19937_64 rng{std::random_device{}()};
    char b[32]; snprintf(b, sizeof b, "%016llx", (unsigned long long)rng());
    return prefix + std::string(b);
}

// The model's tool calls: <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>...</function>\n</tool_call>.
// A parameter is a string unless its schema type says otherwise and the value parses as that JSON.
static std::vector<ToolCall> parse_tool_calls(const std::string& s, const std::vector<Json>& tools, const char* id_prefix) {
    std::vector<ToolCall> out;
    for (size_t p = 0; (p = s.find("<function=", p)) != std::string::npos;) {
        const size_t ne = s.find('>', p);
        if (ne == std::string::npos) break;
        ToolCall tc; tc.name = trim(s.substr(p + 10, ne - p - 10)); tc.args = Json::object(); tc.id = new_call_id(id_prefix);
        size_t fe = s.find("</function>", ne); if (fe == std::string::npos) fe = s.size();
        for (size_t q = ne; (q = s.find("<parameter=", q)) != std::string::npos && q < fe;) {
            const size_t ke = s.find('>', q);
            if (ke == std::string::npos || ke > fe) break;
            const std::string key = trim(s.substr(q + 11, ke - q - 11));
            size_t ve = s.find("</parameter>", ke);
            const size_t nx = s.find("<parameter=", ke);  // a missing </parameter>: the value ends at the next one
            if (ve == std::string::npos || ve > fe) ve = fe;
            if (nx != std::string::npos && nx < ve) ve = nx;
            std::string v = s.substr(ke + 1, ve - ke - 1);
            if (starts_with(v, "\n")) v.erase(0, 1);
            if (ends_with(v, "\n")) v.pop_back();
            const std::string ty = param_type(tools, tc.name, key);
            Json val = Json::string(v);
            if (!ty.empty() && ty != "string") { JsonParser jp(v); Json x = jp.parse(); if (jp.done()) val = x; }
            tc.args.set(key, val);
            q = ve;
        }
        out.push_back(tc);
        p = fe;
    }
    return out;
}

struct Session {
    Engine& eng;
    Tokenizer& tk;
    bool think = false;           // let the model think (<think> ... </think>) instead of an empty think block
    std::vector<int> ctx;         // tokens whose effect is in the engine state, at positions 0..ctx.size()-1
    // Our own replies, keyed by what a client sends back (key_of), so a resent history reuses their exact ids. head is
    // what came after "assistant\n" in the prompt; with thinking, the thinking is in ids, so the history still
    // matches the engine state and only the new turn is prefilled.
    struct Reply { std::string head; std::vector<int> ids; };
    std::map<std::string, Reply> replies;
    int im_start, im_end, eot;
    // Finetunes' variants of the template (chat_template.jinja that `moex import` keeps from the GGUF): Unsloth's fixes
    // (also in Ornith) merge up to two leading system/developer messages with "\n" and drop later ones, and print
    // non-string tool arguments with Python's str(); Ornith keeps every assistant turn's reasoning in the history.
    bool tpl_merge_sys = false, tpl_py_args = false, tpl_keep_reasoning = false;

    Session(Engine& e, Tokenizer& t) : eng(e), tk(t) {
        im_start = tk.id_of("<|im_start|>"); im_end = tk.id_of("<|im_end|>"); eot = tk.id_of("<|endoftext|>");
    }
    void load_template(const std::string& model_dir) {  // none (the downloaded model): Qwen3.6's own template
        FILE* f = fopen((model_dir + "/chat_template.jinja").c_str(), "rb");
        if (!f) return;
        std::string j;
        char b[4096];
        for (size_t n; (n = fread(b, 1, sizeof b, f)) > 0;) j.append(b, n);
        fclose(f);
        tpl_merge_sys = j.find("merged_system") != std::string::npos;
        tpl_py_args = j.find("tool_call.arguments is mapping") != std::string::npos;
        tpl_keep_reasoning = j.find("reasoning_content") != std::string::npos && j.find("loop.index0 > ns.last_query_index") == std::string::npos;
    }

    static std::string key_of(const std::string& content, const std::vector<ToolCall>& calls) {
        std::string k = content;
        for (auto& c : calls) k += "\x1f" + c.id;
        return k;
    }

    // Qwen3.6's chat template. tools: OpenAI form, written one per line as JSON. Our earlier replies go in as the ids
    // we generated (re-tokenizing their text can differ), so a client that resends the whole history extends ctx.
    std::vector<int> render(const std::vector<ChatMsg>& msgs, const std::vector<Json>& tools = {}) {
        std::vector<int> ids;
        auto add = [&](const std::string& s) { auto v = tk.encode(s); ids.insert(ids.end(), v.begin(), v.end()); };
        auto turn = [&](const std::string& s) { ids.push_back(im_start); add(s); ids.push_back(im_end); add("\n"); };
        const bool has_sys = !msgs.empty() && msgs[0].role == "system";
        std::string sys = has_sys ? trim(msgs[0].content) : "";
        int first = has_sys ? 1 : 0;
        if (tpl_merge_sys && has_sys && msgs.size() > 1 && msgs[1].role == "system") { sys += "\n" + trim(msgs[1].content); first = 2; }
        if (!tools.empty()) {
            std::string s = "system\n" + TOOLS_HEAD;
            for (auto& t : tools) s += "\n" + t.dump();
            s += TOOLS_RULES;
            if (!sys.empty()) s += "\n\n" + sys;
            turn(s);
        } else if (has_sys && !(tpl_merge_sys && sys.empty())) turn("system\n" + sys);
        // the last real user query (not wrapped tool responses): assistant turns after it keep their thinking
        int last_query = (int)msgs.size() - 1;
        for (int i = (int)msgs.size() - 1; i >= 0; i--) {
            if (msgs[i].role != "user") continue;
            const std::string c = trim(msgs[i].content);
            if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) { last_query = i; break; }
        }
        for (int i = first; i < (int)msgs.size(); i++) {
            const ChatMsg& m = msgs[i];
            if (tpl_merge_sys && m.role == "system") continue;
            const std::string c = trim(m.content);
            if (m.role == "assistant") {
                ids.push_back(im_start);
                auto it = replies.find(key_of(m.content, m.tool_calls));
                if (it != replies.end()) {  // tokenized exactly as the generation prompt was
                    add("assistant\n" + it->second.head);
                    ids.insert(ids.end(), it->second.ids.begin(), it->second.ids.end());
                } else {
                    std::string rc = m.reasoning, body = c;
                    if (!m.has_reasoning && body.find("</think>") != std::string::npos) {
                        rc = body.substr(0, body.find("</think>"));
                        if (rc.rfind("<think>") != std::string::npos) rc = rc.substr(rc.rfind("<think>") + 7);
                        body = body.substr(body.rfind("</think>") + 8);
                        const size_t nl = body.find_first_not_of('\n');
                        body.erase(0, nl == std::string::npos ? body.size() : nl);
                    }
                    std::string s = i > last_query || tpl_keep_reasoning ? "assistant\n<think>\n" + trim(rc) + "\n</think>\n\n" + body : "assistant\n" + body;
                    for (size_t j = 0; j < m.tool_calls.size(); j++) {
                        const ToolCall& tc = m.tool_calls[j];
                        s += std::string(j ? "\n<tool_call>\n<function=" : !trim(body).empty() ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=") + tc.name + ">\n";
                        for (auto& kv : tc.args.o) s += "<parameter=" + kv.first + ">\n" + arg_text(kv.second) + "\n</parameter>\n";
                        s += "</function>\n</tool_call>";
                    }
                    add(s);
                }
                ids.push_back(im_end); add("\n");
            } else if (m.role == "tool") {
                if (i == 0 || msgs[i - 1].role != "tool") { ids.push_back(im_start); add("user"); }
                add("\n<tool_response>\n" + c + "\n</tool_response>");
                if (i + 1 == (int)msgs.size() || msgs[i + 1].role != "tool") { ids.push_back(im_end); add("\n"); }
            } else turn(m.role + "\n" + c);  // user (and a late system message, which the template would refuse)
        }
        ids.push_back(im_start);
        add("assistant\n" + gen_head());
        return ids;
    }
    // a tool-call argument as the template prints it: strings as they are, else JSON (Python's str() for true, false,
    // null in the finetunes' variant; shortcut: numbers keep their JSON text, which str() would print as 100000.0 for 1e5)
    std::string arg_text(const Json& v) const {
        if (v.t == Json::STR) return v.s;
        if (tpl_py_args && v.t == Json::BOOL) return v.b ? "True" : "False";
        if (tpl_py_args && v.t == Json::NUL) return "None";
        return v.dump();
    }
    static inline const std::string EMPTY_THINK = "<think>\n\n</think>\n\n";
    std::string gen_head() const { return think ? "<think>\n" : EMPTY_THINK; }
    // After generate(): key = key_of(the reply as the client will send it back, its tool calls).
    // Keeps the last MAX_REPLIES (a long-running server must not grow forever; an evicted reply only costs a re-prefill).
    static constexpr size_t MAX_REPLIES = 512;
    std::deque<std::string> reply_order;
    void remember(const std::string& key, const std::vector<int>& out) {
        if (!replies.count(key)) reply_order.push_back(key);
        replies[key] = {gen_head(), out};
        while (replies.size() > MAX_REPLIES) { replies.erase(reply_order.front()); reply_order.pop_front(); }
    }

    // Runs `prompt` and streams up to max_new generated tokens to on_tok (return false to stop early). Stops at
    // <|im_end|> / <|endoftext|> (not emitted, not fed). Returns the generated ids.
    std::vector<int> generate(const std::vector<int>& prompt, int max_new, const std::function<bool(int)>& on_tok) {
        size_t common = 0;
        while (common < ctx.size() && common < prompt.size() && ctx[common] == prompt[common]) common++;
        if ((int)prompt.size() >= eng.ctx) { fprintf(stderr, "[session] prompt (%zu tokens) exceeds --ctx %d\n", prompt.size(), eng.ctx); return {}; }
        int next;
        if (common > 0 && common == ctx.size() && common < prompt.size()) {  // extends what the engine holds
            next = eng.prefill(prompt.data() + common, (int)(prompt.size() - common), (int)common);
        } else {  // a new conversation (/reset leaves ctx empty) or a changed history: start over from a clean state
                  // (the DeltaNet state cannot be rewound, and the engine may still hold an earlier conversation)
            eng.reset();
            next = eng.prefill(prompt.data(), (int)prompt.size(), 0);
        }
        ctx = prompt;
        std::vector<int> out;
        std::deque<int> q{next};  // generated, not yet emitted; with MTP a round can yield two
        while (!q.empty() && (int)out.size() < max_new) {
            const int t = q.front(); q.pop_front();
            if (t == im_end || t == eot) break;
            out.push_back(t);
            if (!on_tok(t) || (int)ctx.size() + 3 >= eng.ctx) break;
            if (!q.empty()) continue;  // t was already fed (an accepted MTP draft)
            const int pos = (int)ctx.size();
            ctx.push_back(t);
            if (eng.has_mtp) {
                auto ts = eng.mtp_round(t, pos);
                ctx.insert(ctx.end(), ts.begin(), ts.end() - 1);  // accepted drafts are fed; the last token is not
                q.insert(q.end(), ts.begin(), ts.end());
            } else q.push_back(eng.step(t, pos, true));
        }
        return out;
    }
};

// Splits a thinking reply into reasoning and answer as it streams: an optional "<think>" at the start, the
// reasoning, "</think>", then the answer (whitespace around both trimmed at their start). Holds back a tail that
// could still become a tag. Without thinking everything is answer.
struct ThinkSplit {
    static inline const std::string OPEN = "<think>", CLOSE = "</think>";
    bool in_think, opened = false, lead = true;
    std::string buf;
    explicit ThinkSplit(bool think) : in_think(think) {}
    static size_t partial_tail(const std::string& s, const std::string& tag) {  // longest proper prefix of tag ending s
        for (size_t n = std::min(s.size(), tag.size() - 1); n > 0; n--)
            if (!s.compare(s.size() - n, n, tag, 0, n)) return n;
        return 0;
    }
    bool trim_lead(bool last) {  // drops leading whitespace; false = nothing else yet, wait for more
        if (!lead) return true;
        size_t i = 0;
        while (i < buf.size() && strchr(" \t\r\n", buf[i])) i++;
        buf.erase(0, i);
        if (buf.empty() && !last) return false;
        lead = false;
        return true;
    }
    void push(const std::string& s, std::string& reasoning, std::string& answer, bool last) {
        buf += s;
        if (in_think) {
            if (!opened) {
                if (!trim_lead(last)) return;
                if (buf.size() < OPEN.size() && !last && !OPEN.compare(0, buf.size(), buf)) return;  // may still be "<think>"
                if (!buf.compare(0, OPEN.size(), OPEN)) buf.erase(0, OPEN.size());
                opened = true; lead = true;
            }
            if (!trim_lead(last)) return;
            size_t k = buf.find(CLOSE);
            if (k == std::string::npos) {
                size_t keep = last ? 0 : partial_tail(buf, CLOSE);
                reasoning += buf.substr(0, buf.size() - keep);
                buf.erase(0, buf.size() - keep);
                return;
            }
            reasoning += buf.substr(0, k);
            buf.erase(0, k + CLOSE.size());
            in_think = false; lead = true;
        }
        if (lead && opened && !trim_lead(last)) return;
        answer += buf;
        buf.clear();
    }
};

// Splits an answer into plain content (streamable) and the tool-call text from the first <tool_call> on.
struct CallSplit {
    static inline const std::string TAG = "<tool_call>";
    std::string buf, calls;
    bool in_calls = false;
    void push(const std::string& s, std::string& content, bool last) {
        if (in_calls) { calls += s; return; }
        buf += s;
        size_t k = buf.find(TAG);
        if (k != std::string::npos) { content += buf.substr(0, k); calls = buf.substr(k); buf.clear(); in_calls = true; return; }
        size_t keep = last ? 0 : ThinkSplit::partial_tail(buf, TAG);
        content += buf.substr(0, buf.size() - keep);
        buf.erase(0, buf.size() - keep);
    }
};

// Streams token bytes as text, holding back an incomplete UTF-8 sequence until its last byte arrives.
struct Utf8Stream {
    std::string held;
    std::string push(const std::string& bytes) {
        held += bytes;
        size_t cut = held.size(), k = held.size();
        while (k > 0 && k + 4 > held.size() && ((unsigned char)held[k - 1] & 0xC0) == 0x80) k--;  // continuation bytes
        if (k > 0) {
            unsigned char lead = held[k - 1];
            int need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
            if ((int)(held.size() - (k - 1)) < need) cut = k - 1;
        }
        std::string out = held.substr(0, cut);
        held.erase(0, cut);
        return out;
    }
};
