// HTTP server (Winsock / POSIX sockets, no dependencies) with two APIs over the same engine:
//   OpenAI-compatible:    POST /v1/chat/completions (tools, streaming), GET /v1/models
//   Anthropic-compatible: POST /v1/messages (tools, thinking, streaming), POST /v1/messages/count_tokens
// plus GET /health and GET /api (help). One request at a time; listens on 127.0.0.1 unless told otherwise.
// Included by moex.cu after Session.
#pragma once
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <csignal>
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define closesocket close
#endif
#include <algorithm>
#include <ctime>
#include <functional>
#include <string>
#include <vector>
#include "json.h"
#pragma comment(lib, "ws2_32.lib")

// --host: an IPv4 address (four numbers 0-255) or "localhost". Anything else is refused: a typo must not end up
// binding every interface (an unparsed address is 0.0.0.0).
static bool parse_host(const std::string& h, uint8_t out[4]) {
    if (h == "localhost") { out[0] = 127; out[1] = 0; out[2] = 0; out[3] = 1; return true; }
    int n = 0; size_t p = 0;
    while (n < 4) {
        size_t q = p;
        while (q < h.size() && isdigit((unsigned char)h[q]) && q - p < 3) q++;
        if (q == p || (q < h.size() && isdigit((unsigned char)h[q]))) return false;
        int v = atoi(h.substr(p, q - p).c_str());
        if (v > 255) return false;
        out[n++] = (uint8_t)v;
        if (n < 4) { if (q >= h.size() || h[q] != '.') return false; p = q + 1; } else p = q;
    }
    return p == h.size();
}

// ---- HTTP ----
static bool send_all(SOCKET c, const std::string& s) {
    for (size_t p = 0; p < s.size();) { int n = send(c, s.data() + p, (int)(s.size() - p), 0); if (n <= 0) return false; p += n; }
    return true;
}
// CORS headers for the current request: only an allowed Origin is echoed back (never "*"), so a web page from
// another site cannot read moex's replies. Set per request by serve(); one request at a time.
static std::string CORS;
// A header's value (name lowercase), "" if absent.
static std::string header_of(const std::string& head, const std::string& name) {
    std::string low = head;
    for (auto& ch : low) ch = (char)tolower(ch);
    size_t k = low.find("\r\n" + name + ":");
    if (k == std::string::npos) return "";
    k += name.size() + 3;
    size_t e = head.find("\r\n", k);
    return trim(head.substr(k, (e == std::string::npos ? head.size() : e) - k));
}
// "http://localhost:3000" -> "localhost"; "[::1]:8765" -> "::1"; "127.0.0.1:8765" -> "127.0.0.1"
static std::string host_part(std::string s) {
    size_t sch = s.find("://");
    if (sch != std::string::npos) s = s.substr(sch + 3);
    s = s.substr(0, s.find('/'));
    if (!s.empty() && s[0] == '[') return s.substr(1, s.find(']') - 1);
    return s.substr(0, s.find(':'));
}
// localhost, *.localhost, ::1 or a real IPv4 127.x.x.x (a name like 127.0.0.1.evil.com is someone's domain)
static bool is_loopback_name(const std::string& h) {
    std::string l = h; for (auto& ch : l) ch = (char)tolower(ch);
    uint8_t ip[4];
    return l == "localhost" || l == "::1" || ends_with(l, ".localhost") || (parse_host(l, ip) && ip[0] == 127);
}
static void respond(SOCKET c, int code, const std::string& type, const std::string& body) {
    const char* msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 403 ? "Forbidden" : code == 404 ? "Not Found" : "Error";
    send_all(c, "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + type + "\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n" + CORS + "Connection: close\r\n\r\n" + body);
}
static void start_sse(SOCKET c) {
    send_all(c, std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n") + CORS + "Connection: close\r\n\r\n");
}

// MOEX_DUMP_PROMPT=file: write each rendered prompt's ids there (tools/check_template.py compares them with the template)
static void dump_prompt(const std::vector<int>& ids) {
    const char* f = getenv("MOEX_DUMP_PROMPT");
    if (!f) return;
    if (FILE* o = fopen(f, "wb")) { for (size_t i = 0; i < ids.size(); i++) fprintf(o, i ? ",%d" : "%d", ids[i]); fclose(o); }
}

// ---- one generation, whichever API asked ----
struct GenReq {
    std::vector<ChatMsg> msgs;
    std::vector<Json> tools;  // OpenAI form: {"type":"function","function":{"name","description","parameters"}}
    std::vector<std::string> stops;
    int max_new = 0;  // 0: up to the end of the context
    bool think = false;
    Engine::Sampling sp;
    const char* id_prefix = "call_";
};
struct GenOut {
    std::string reasoning, content, stop_seq, error;
    std::vector<ToolCall> calls;
    size_t prompt_n = 0, gen_n = 0;
    bool hit_stop = false, length = false;
};

// Thinking goes to on_reason, the answer to on_content (both return false once the client is gone). Tool calls are
// parsed at the end; the answer text before them streams, minus trailing whitespace. Stop strings apply to the answer.
static GenOut run_request(Session& ss, Tokenizer& tk, const GenReq& q, const std::function<void(size_t)>& on_start,
                          const std::function<bool(const std::string&)>& on_reason, const std::function<bool(const std::string&)>& on_content) {
    GenOut g;
    ss.think = q.think;
    ss.eng.samp = q.sp;
    if (q.sp.seed) ss.eng.rng.seed(q.sp.seed);
    auto prompt = ss.render(q.msgs, q.tools);
    dump_prompt(prompt);
    g.prompt_n = prompt.size();
    if ((int)prompt.size() + 1 >= ss.eng.ctx) {
        g.error = "the prompt is " + std::to_string(prompt.size()) + " tokens, more than the context (" + std::to_string(ss.eng.ctx) +
                  "): restart moex serve with a larger --ctx, e.g. --ctx 32768";
        fprintf(stderr, "[serve] %s\n", g.error.c_str());
        return g;
    }
    on_start(prompt.size());
    const bool tools = !q.tools.empty();
    Utf8Stream us; ThinkSplit split(q.think); CallSplit cs;
    bool alive = true; size_t sent = 0;
    size_t hold = 0; for (auto& s : q.stops) hold = std::max(hold, s.size() - 1);
    auto feed = [&](const std::string& piece, bool last) {
        std::string rd, ad;
        split.push(piece, rd, ad, last);
        if (!rd.empty()) { g.reasoning += rd; if (alive) alive = on_reason(rd); }
        if (tools) cs.push(ad, g.content, last); else g.content += ad;
        if (!g.hit_stop)
            for (auto& s : q.stops) {
                size_t k = g.content.find(s);
                if (k != std::string::npos) { g.content.resize(k); g.hit_stop = true; g.stop_seq = s; break; }
            }
        size_t can;
        if (cs.in_calls) {  // the text before a tool call ends here: drop its trailing whitespace
            while (g.content.size() > sent && strchr(" \t\r\n", g.content.back())) g.content.pop_back();
            can = g.content.size();
        } else if (g.hit_stop || last) can = g.content.size();
        else {
            can = g.content.size() > hold ? g.content.size() - hold : 0;  // a tail that could still become a stop string
            if (tools) while (can > sent && strchr(" \t\r\n", g.content[can - 1])) can--;  // whitespace that may precede a tool call
        }
        if (alive && can > sent) { alive = on_content(g.content.substr(sent, can - sent)); sent = can; }
    };
    const int max_new = q.max_new > 0 ? q.max_new : ss.eng.ctx - (int)prompt.size() - 1;
    const double t0 = now();
    auto out = ss.generate(prompt, max_new, [&](int t) {
        feed(us.push(tk.piece(t)), false);
        return alive && !g.hit_stop;
    });
    if (!g.hit_stop) feed(us.held, true);
    if (cs.in_calls) g.calls = parse_tool_calls(cs.calls, q.tools, q.id_prefix);
    g.gen_n = out.size();
    g.length = !g.hit_stop && ((int)out.size() >= max_new || (int)(prompt.size() + out.size()) + 4 >= ss.eng.ctx);
    ss.remember(Session::key_of(g.content, g.calls), out);
    ss.eng.save_hot();
    fprintf(stderr, "[serve] %zu prompt tokens, %zu generated%s, %.1f s, %.1f tok/s overall\n", prompt.size(), out.size(),
            g.calls.empty() ? "" : (", " + std::to_string(g.calls.size()) + " tool call(s)").c_str(), now() - t0,
            out.size() / std::max(now() - t0, 1e-9));
    return g;
}

// Sampling defaults: Qwen's recommendations for thinking / not thinking, unless the request says.
static Engine::Sampling sampling_of(const Json& r, bool think) {
    Engine::Sampling sp;
    sp.temp = (float)r.num("temperature", think ? 1.0 : 0.7);
    sp.top_p = (float)r.num("top_p", think ? 0.95 : 0.8);
    sp.top_k = (int)r.num("top_k", 20);
    sp.rep = (float)r.num("repetition_penalty", 1.0);
    sp.seed = (uint64_t)r.num("seed", 0);
    return sp;
}
static std::string text_of(const Json* co) {  // a string, or the text parts of a content array
    std::string s;
    if (co && co->t == Json::STR) s = co->s;
    else if (co && co->t == Json::ARR)
        for (auto& part : co->a) {
            if (part.t == Json::STR) s += part.s;
            else if (auto tx = part.get("text")) if (tx->t == Json::STR) s += tx->s;
        }
    return s;
}
static std::vector<std::string> strings_of(const Json* st) {
    std::vector<std::string> v;
    if (st && st->t == Json::STR) v.push_back(st->s);
    else if (st && st->t == Json::ARR) for (auto& x : st->a) if (x.t == Json::STR) v.push_back(x.s);
    v.erase(std::remove(v.begin(), v.end(), std::string()), v.end());
    return v;
}
static std::string rand_id(const char* prefix) { return new_call_id(prefix); }

// ---- OpenAI: POST /v1/chat/completions ----
static std::string usage_of(const GenOut& g) {
    return "{\"prompt_tokens\":" + std::to_string(g.prompt_n) + ",\"completion_tokens\":" + std::to_string(g.gen_n) +
           ",\"total_tokens\":" + std::to_string(g.prompt_n + g.gen_n) + "}";
}
static bool openai_request(const Json& r, bool think_default, GenReq& q, std::string& err) {
    const Json* msgs = r.get("messages");
    if (!msgs || msgs->t != Json::ARR || msgs->a.empty()) { err = "messages missing or invalid JSON"; return false; }
    for (auto& x : msgs->a) {
        ChatMsg cm;
        cm.role = x.str("role", "user");
        if (cm.role == "developer") cm.role = "system";
        cm.content = text_of(x.get("content"));
        for (const char* k : {"reasoning_content", "reasoning"})
            if (auto rc = x.get(k)) if (rc->t == Json::STR) { cm.reasoning = rc->s; cm.has_reasoning = true; break; }
        if (auto tcs = x.get("tool_calls")) if (tcs->t == Json::ARR)
            for (auto& t : tcs->a) {
                ToolCall tc; tc.id = t.str("id"); tc.args = Json::object();
                const Json* f = t.get("function"); if (!f) continue;
                tc.name = f->str("name");
                if (auto a = f->get("arguments")) {
                    if (a->t == Json::OBJ) tc.args = *a;
                    else if (a->t == Json::STR) { JsonParser jp(a->s); Json v = jp.parse(); if (jp.done() && v.t == Json::OBJ) tc.args = v; }
                }
                cm.tool_calls.push_back(tc);
            }
        q.msgs.push_back(cm);
    }
    if (auto t = r.get("tools")) if (t->t == Json::ARR) for (auto& x : t->a) if (x.get("function")) q.tools.push_back(x);
    if (auto tc = r.get("tool_choice")) if (tc->t == Json::STR && tc->s == "none") q.tools.clear();
    // thinking: chat_template_kwargs.enable_thinking, else reasoning_effort ("none" = off), else `serve --think`
    auto kw = r.get("chat_template_kwargs");
    auto re = r.get("reasoning_effort");
    q.think = kw && kw->get("enable_thinking") ? kw->boolean("enable_thinking", think_default)
            : re && re->t == Json::STR ? re->s != "none"
            : think_default;
    q.max_new = (int)r.num("max_completion_tokens", r.num("max_tokens", 0));
    q.stops = strings_of(r.get("stop"));
    q.sp = sampling_of(r, q.think);
    q.id_prefix = "call_";
    return true;
}
static Json openai_tool_call(const ToolCall& tc, int index) {
    Json f = Json::object(); f.set("name", Json::string(tc.name)).set("arguments", Json::string(tc.args.dump()));
    Json j = Json::object();
    if (index >= 0) { Json ix; ix.t = Json::NUM; ix.s = std::to_string(index); j.set("index", ix); }
    j.set("id", Json::string(tc.id)).set("type", Json::string("function")).set("function", f);
    return j;
}
static void openai_serve(SOCKET c, Session& ss, Tokenizer& tk, const Json& r, bool think_default, const std::string& model_id) {
    GenReq q; std::string err;
    if (!openai_request(r, think_default, q, err)) { respond(c, 400, "application/json", "{\"error\":{\"message\":\"" + jesc(err) + "\"}}"); return; }
    const bool stream = r.boolean("stream", false);
    const std::string id = rand_id("chatcmpl-");
    const std::string created = std::to_string((long long)time(nullptr));
    auto chunk = [&](const std::string& delta, const std::string& finish) {
        return "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + created + ",\"model\":\"" +
               jesc(model_id) + "\",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":" +
               (finish.empty() ? "null" : "\"" + finish + "\"") + "}]}\n\n";
    };
    auto field = [&](const char* k, const std::string& v) { return "{\"" + std::string(k) + "\":\"" + jesc(v) + "\"}"; };
    GenOut g = run_request(ss, tk, q,
        [&](size_t) { if (stream) { start_sse(c); send_all(c, chunk("{\"role\":\"assistant\",\"content\":\"\"}", "")); } },
        [&](const std::string& d) { return !stream || send_all(c, chunk(field("reasoning_content", d), "")); },
        [&](const std::string& d) { return !stream || send_all(c, chunk(field("content", d), "")); });
    if (!g.error.empty()) { respond(c, 400, "application/json", "{\"error\":{\"message\":\"" + jesc(g.error) + "\"}}"); return; }
    const std::string finish = !g.calls.empty() ? "tool_calls" : g.length ? "length" : "stop";
    if (stream) {
        for (size_t i = 0; i < g.calls.size(); i++)
            send_all(c, chunk("{\"tool_calls\":[" + openai_tool_call(g.calls[i], (int)i).dump() + "]}", ""));
        send_all(c, chunk("{}", finish));
        if (const Json* so = r.get("stream_options")) if (so->boolean("include_usage", false))  // a last chunk with the counts
            send_all(c, "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + created + ",\"model\":\"" +
                        jesc(model_id) + "\",\"choices\":[],\"usage\":" + usage_of(g) + "}\n\n");
        send_all(c, "data: [DONE]\n\n");
        return;
    }
    std::string msg = "{\"role\":\"assistant\",\"content\":" + (g.content.empty() && !g.calls.empty() ? std::string("null") : "\"" + jesc(g.content) + "\"");
    if (!g.reasoning.empty()) msg += ",\"reasoning_content\":\"" + jesc(g.reasoning) + "\"";
    if (!g.calls.empty()) {
        msg += ",\"tool_calls\":[";
        for (size_t i = 0; i < g.calls.size(); i++) msg += (i ? "," : "") + openai_tool_call(g.calls[i], -1).dump();
        msg += "]";
    }
    msg += "}";
    respond(c, 200, "application/json",
            "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + created + ",\"model\":\"" + jesc(model_id) +
            "\",\"choices\":[{\"index\":0,\"message\":" + msg + ",\"finish_reason\":\"" + finish + "\"}],\"usage\":" + usage_of(g) + "}");
}

// ---- Anthropic: POST /v1/messages ----
static bool anthropic_request(const Json& r, bool think_default, GenReq& q, std::string& err) {
    const Json* msgs = r.get("messages");
    if (!msgs || msgs->t != Json::ARR || msgs->a.empty()) { err = "messages missing or invalid JSON"; return false; }
    if (auto s = r.get("system")) { std::string t = text_of(s); if (!t.empty()) q.msgs.push_back({"system", t}); }
    for (auto& x : msgs->a) {
        const std::string role = x.str("role", "user");
        const Json* co = x.get("content");
        if (!co || co->t != Json::ARR) { q.msgs.push_back({role, text_of(co)}); continue; }
        if (role == "assistant") {
            ChatMsg cm; cm.role = "assistant";
            for (auto& b : co->a) {
                const std::string ty = b.str("type");
                if (ty == "text") cm.content += b.str("text");
                else if (ty == "thinking") { cm.reasoning += b.str("thinking"); cm.has_reasoning = true; }
                else if (ty == "tool_use") {
                    ToolCall tc; tc.id = b.str("id"); tc.name = b.str("name"); tc.args = Json::object();
                    if (auto in = b.get("input")) if (in->t == Json::OBJ) tc.args = *in;
                    cm.tool_calls.push_back(tc);
                }
            }
            q.msgs.push_back(cm);
        } else {  // tool results become tool messages (in order), the rest of the text one user message after them
            std::string text;
            for (auto& b : co->a) {
                const std::string ty = b.str("type");
                if (ty == "tool_result") {
                    std::string t = text_of(b.get("content"));
                    if (b.boolean("is_error", false)) t = "Error: " + t;
                    q.msgs.push_back({"tool", t});
                } else if (ty == "text") text += b.str("text");
            }
            if (!trim(text).empty() || q.msgs.empty() || q.msgs.back().role != "tool") q.msgs.push_back({"user", text});
        }
    }
    if (auto t = r.get("tools")) if (t->t == Json::ARR)
        for (auto& x : t->a) {
            if (x.str("name").empty() || !x.get("input_schema")) continue;  // server tools (web search, ...) can't run here
            Json f = Json::object();
            f.set("name", Json::string(x.str("name")));
            if (auto d = x.get("description")) f.set("description", *d);
            f.set("parameters", *x.get("input_schema"));
            Json tool = Json::object(); tool.set("type", Json::string("function")).set("function", f);
            q.tools.push_back(tool);
        }
    if (auto tc = r.get("tool_choice")) if (tc->str("type") == "none") q.tools.clear();
    const Json* th = r.get("thinking");
    q.think = th && th->get("type") ? th->str("type") != "disabled" : think_default;
    q.max_new = (int)r.num("max_tokens", 4096);
    q.stops = strings_of(r.get("stop_sequences"));
    q.sp = sampling_of(r, q.think);
    q.id_prefix = "toolu_";
    return true;
}
static std::string anthropic_error(const std::string& m) {
    return "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"" + jesc(m) + "\"}}";
}
static void anthropic_serve(SOCKET c, Session& ss, Tokenizer& tk, const Json& r, bool think_default, const std::string& model_id) {
    GenReq q; std::string err;
    if (!anthropic_request(r, think_default, q, err)) { respond(c, 400, "application/json", anthropic_error(err)); return; }
    const bool stream = r.boolean("stream", false);
    const std::string id = rand_id("msg_"), model = r.str("model", model_id);
    auto ev = [&](const char* type, const std::string& data) { return send_all(c, std::string("event: ") + type + "\ndata: " + data + "\n\n"); };
    int block = -1; std::string open;  // the content block being streamed: its index and type
    auto close_block = [&] {
        if (!open.empty()) ev("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":" + std::to_string(block) + "}");
        open.clear();
    };
    auto delta = [&](const std::string& type, const std::string& d) {  // type: thinking or text
        if (open != type) {
            close_block();
            open = type; block++;
            ev("content_block_start", "{\"type\":\"content_block_start\",\"index\":" + std::to_string(block) +
                ",\"content_block\":{\"type\":\"" + type + "\",\"" + type + "\":\"\"}}");
        }
        return ev("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(block) +
                  ",\"delta\":{\"type\":\"" + type + "_delta\",\"" + type + "\":\"" + jesc(d) + "\"}}");
    };
    GenOut g = run_request(ss, tk, q,
        [&](size_t n) {
            if (!stream) return;
            start_sse(c);
            ev("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"" + id + "\",\"type\":\"message\",\"role\":\"assistant\","
               "\"content\":[],\"model\":\"" + jesc(model) + "\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":" +
               std::to_string(n) + ",\"output_tokens\":0}}}");
        },
        [&](const std::string& d) { return !stream || delta("thinking", d); },
        [&](const std::string& d) { return !stream || delta("text", d); });
    if (!g.error.empty()) { respond(c, 400, "application/json", anthropic_error(g.error)); return; }
    const std::string stop = !g.calls.empty() ? "tool_use" : g.hit_stop ? "stop_sequence" : g.length ? "max_tokens" : "end_turn";
    const std::string stop_seq = g.hit_stop ? "\"" + jesc(g.stop_seq) + "\"" : "null";
    auto tool_block = [&](const ToolCall& tc, bool with_input) {
        return "{\"type\":\"tool_use\",\"id\":\"" + tc.id + "\",\"name\":\"" + jesc(tc.name) + "\",\"input\":" + (with_input ? tc.args.dump() : "{}") + "}";
    };
    if (stream) {
        for (auto& tc : g.calls) {
            close_block();
            block++; open = "tool_use";
            ev("content_block_start", "{\"type\":\"content_block_start\",\"index\":" + std::to_string(block) + ",\"content_block\":" + tool_block(tc, false) + "}");
            ev("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(block) +
               ",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"" + jesc(tc.args.dump()) + "\"}}");
        }
        close_block();
        ev("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"" + stop + "\",\"stop_sequence\":" + stop_seq +
           "},\"usage\":{\"output_tokens\":" + std::to_string(g.gen_n) + "}}");
        ev("message_stop", "{\"type\":\"message_stop\"}");
        return;
    }
    std::string content = "[";
    auto add = [&](const std::string& b) { content += (content.size() > 1 ? "," : "") + b; };
    if (!g.reasoning.empty()) add("{\"type\":\"thinking\",\"thinking\":\"" + jesc(g.reasoning) + "\",\"signature\":\"\"}");
    if (!g.content.empty()) add("{\"type\":\"text\",\"text\":\"" + jesc(g.content) + "\"}");
    for (auto& tc : g.calls) add(tool_block(tc, true));
    content += "]";
    respond(c, 200, "application/json",
            "{\"id\":\"" + id + "\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"" + jesc(model) + "\",\"content\":" + content +
            ",\"stop_reason\":\"" + stop + "\",\"stop_sequence\":" + stop_seq + ",\"usage\":{\"input_tokens\":" + std::to_string(g.prompt_n) +
            ",\"output_tokens\":" + std::to_string(g.gen_n) + "}}");
}

// ---- OpenAI legacy: POST /v1/completions (raw text in, raw continuation out; editor autocomplete uses it) ----
static void completions_serve(SOCKET c, Session& ss, Tokenizer& tk, const Json& r, const std::string& model_id) {
    const Json* pj = r.get("prompt");
    std::string prompt_text = pj && pj->t == Json::ARR && !pj->a.empty() ? pj->a[0].s : pj && pj->t == Json::STR ? pj->s : "";
    if (prompt_text.empty()) { respond(c, 400, "application/json", "{\"error\":{\"message\":\"prompt missing\"}}"); return; }
    std::vector<int> prompt = tk.encode(prompt_text);
    if ((int)prompt.size() + 1 >= ss.eng.ctx) {
        respond(c, 400, "application/json", "{\"error\":{\"message\":\"the prompt is " + std::to_string(prompt.size()) +
                " tokens, more than the context (" + std::to_string(ss.eng.ctx) + "): restart moex serve with a larger --ctx\"}}");
        return;
    }
    const bool stream = r.boolean("stream", false), echo = r.boolean("echo", false);
    int max_new = (int)r.num("max_tokens", 16);  // the legacy API's default
    if (max_new <= 0 || max_new > ss.eng.ctx - (int)prompt.size() - 1) max_new = ss.eng.ctx - (int)prompt.size() - 1;
    const auto stops = strings_of(r.get("stop"));
    ss.eng.samp = sampling_of(r, false);
    if (ss.eng.samp.seed) ss.eng.rng.seed(ss.eng.samp.seed);
    const std::string id = rand_id("cmpl-"), created = std::to_string((long long)time(nullptr));
    auto chunk = [&](const std::string& text, const std::string& finish) {
        return "data: {\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created + ",\"model\":\"" + jesc(model_id) +
               "\",\"choices\":[{\"text\":\"" + jesc(text) + "\",\"index\":0,\"logprobs\":null,\"finish_reason\":" +
               (finish.empty() ? "null" : "\"" + finish + "\"") + "}]}\n\n";
    };
    if (stream) { start_sse(c); if (echo) send_all(c, chunk(prompt_text, "")); }
    Utf8Stream us; std::string text; bool alive = true, hit_stop = false; size_t sent = 0;
    size_t hold = 0; for (auto& s : stops) hold = std::max(hold, s.size() - 1);
    auto feed = [&](const std::string& piece, bool last) {
        text += piece;
        for (auto& s : stops) { size_t k = text.find(s); if (k != std::string::npos) { text.resize(k); hit_stop = true; break; } }
        size_t can = hit_stop || last ? text.size() : (text.size() > hold ? text.size() - hold : 0);
        if (stream && alive && can > sent) { alive = send_all(c, chunk(text.substr(sent, can - sent), "")); sent = can; }
    };
    auto out = ss.generate(prompt, max_new, [&](int t) { feed(us.push(tk.piece(t)), false); return alive && !hit_stop; });
    if (!hit_stop) feed(us.held, true);
    ss.eng.save_hot();
    const std::string finish = !hit_stop && (int)out.size() >= max_new ? "length" : "stop";
    const std::string usage = "{\"prompt_tokens\":" + std::to_string(prompt.size()) + ",\"completion_tokens\":" + std::to_string(out.size()) +
                              ",\"total_tokens\":" + std::to_string(prompt.size() + out.size()) + "}";
    fprintf(stderr, "[serve] completions: %zu prompt tokens, %zu generated\n", prompt.size(), out.size());
    if (stream) {
        send_all(c, chunk("", finish));
        if (const Json* so = r.get("stream_options")) if (so->boolean("include_usage", false))
            send_all(c, "data: {\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created + ",\"model\":\"" +
                        jesc(model_id) + "\",\"choices\":[],\"usage\":" + usage + "}\n\n");
        send_all(c, "data: [DONE]\n\n");
        return;
    }
    respond(c, 200, "application/json",
            "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + created + ",\"model\":\"" + jesc(model_id) +
            "\",\"choices\":[{\"text\":\"" + jesc((echo ? prompt_text : "") + text) + "\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"" +
            finish + "\"}],\"usage\":" + usage + "}");
}

// Who may call: with moex on loopback, the Host header must name loopback too (a DNS-rebinding page names its own
// domain). A browser request (it carries Origin) is allowed from loopback pages and from --allow-origin; requests
// without Origin (curl, SDKs, Open WebUI's backend) are not affected. Returns "" if allowed, else the reason.
static std::string check_caller(const std::string& head, bool loopback, const std::vector<std::string>& allow_origins,
                                std::string& cors) {
    cors.clear();
    const std::string hosth = header_of(head, "host"), origin = header_of(head, "origin");
    if (loopback && !hosth.empty() && !is_loopback_name(host_part(hosth)))
        return "Host '" + hosth + "' is not this computer (moex serve listens on localhost only)";
    if (origin.empty()) return "";
    bool ok = is_loopback_name(host_part(origin));
    for (auto& a : allow_origins) ok |= a == "*" || a == origin;
    if (!ok) return "requests from web pages at " + origin + " are not allowed (start moex serve with --allow-origin " + origin + " to allow them)";
    cors = "Access-Control-Allow-Origin: " + origin + "\r\nVary: Origin\r\nAccess-Control-Allow-Headers: *\r\n"
           "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    return "";
}

static void serve(Session& ss, Tokenizer& tk, const std::string& host, int port, const std::string& model_id,
                  const std::vector<std::string>& allow_origins = {}) {
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);  // a client that hangs up mid-stream must not end the server
#endif
    const bool think_default = ss.think;  // `moex serve --think`
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    uint8_t ip[4];
    if (!parse_host(host, ip)) { fprintf(stderr, "[serve] --host %s is not an IPv4 address (e.g. 127.0.0.1, or 0.0.0.0 for your network)\n", host.c_str()); return; }
    memcpy(&a.sin_addr, ip, 4);
    const bool loopback = ip[0] == 127;
    int yes = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 16)) { fprintf(stderr, "[serve] cannot listen on %s:%d (port in use? choose another with --port)\n", host.c_str(), port); return; }
    const std::string base = "http://" + std::string(host == "0.0.0.0" ? "127.0.0.1" : host) + ":" + std::to_string(port);
    const std::string ps_example = "Invoke-RestMethod " + base + "/v1/chat/completions -Method Post -ContentType application/json "
                                   "-Body '{\"model\":\"" + model_id + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'";
    const std::string py_example = "from openai import OpenAI; c = OpenAI(base_url=\"" + base + "/v1\", api_key=\"none\"); "
                                   "print(c.chat.completions.create(model=\"" + model_id + "\", messages=[{\"role\": \"user\", \"content\": \"Hello!\"}]).choices[0].message.content)";
    // GET /api (and /): what this server offers and how to call it
    const std::string api_help = "{\"server\":\"moex\",\"model\":\"" + jesc(model_id) + "\",\"base_url\":\"" + jesc(base) + "/v1\",\"api_key\":\"any\","
        "\"endpoints\":{\"POST /v1/chat/completions\":\"OpenAI chat completions; \\\"stream\\\": true for streaming. Also: tools (tool_calls back), "
        "temperature, top_p, top_k, seed, stop, max_tokens; thinking: chat_template_kwargs.enable_thinking or reasoning_effort (default: serve --think), "
        "sent as reasoning_content, the answer as content\",\"POST /v1/completions\":\"OpenAI legacy completions: raw prompt, no chat template "
        "(editor autocomplete); max_tokens (default 16), stop, echo, stream\",\"POST /v1/messages\":\"Anthropic Messages API: system, tools (tool_use / tool_result), "
        "thinking, stop_sequences, streaming\",\"POST /v1/messages/count_tokens\":\"prompt size in tokens\",\"GET /v1/models\":\"the model id\","
        "\"GET /health\":\"{\\\"status\\\":\\\"ok\\\"}\",\"GET /api\":\"this page\"},\"examples\":{\"powershell\":\"" + jesc(ps_example) +
        "\",\"python\":\"" + jesc(py_example) + "\"}}";
    fprintf(stderr,
            "\n[serve] moex is serving %s - OpenAI- and Anthropic-compatible API\n"
            "  Base URL:   %s/v1   (model \"%s\", any API key)\n"
            "  Endpoints:  POST /v1/chat/completions   POST /v1/completions   POST /v1/messages   GET /v1/models   GET /health   GET /api (help)\n"
            "  Try it in PowerShell:\n    %s\n"
            "  Python (pip install openai):\n    %s\n"
            "  Open WebUI, editors and other apps: add an OpenAI-compatible connection with the base URL above.\n"
            "  Apps for Anthropic's API: base URL %s. Coding agents need a long context: --ctx 32768 (now %d).\n"
            "  %s\n"
            "  Ctrl+C stops the server.\n\n",
            model_id.c_str(), base.c_str(), model_id.c_str(), ps_example.c_str(), py_example.c_str(), base.c_str(), ss.eng.ctx,
            loopback ? "Only this PC can connect (--host 0.0.0.0 opens it to your network). Web pages: only localhost ones (--allow-origin URL for others)."
                     : "Other devices on your network can connect too (no API key: only on networks you trust).");
    while (true) {
        SOCKET c = accept(ls, nullptr, nullptr);
        if (c == INVALID_SOCKET) continue;
        // One request at a time, so a client that stalls must not hold the server: 15 s per read, 60 s per write
        // (a stream to a client that stopped reading).
#ifdef _WIN32
        DWORD rto = 15000, sto = 60000;
#else
        timeval rto{15, 0}, sto{60, 0};
#endif
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, (const char*)&rto, sizeof rto);
        setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, (const char*)&sto, sizeof sto);
        constexpr long long MAX_BODY = 64ll << 20, MAX_HEAD = 1 << 20;
        std::string req; char buf[65536];
        size_t hdr_end = std::string::npos; long long clen = 0;
        int refuse = 0;  // 413 body too large, 431 headers too large, 408 incomplete
        while (true) {  // headers, then Content-Length bytes of body
            int n = recv(c, buf, sizeof buf, 0);
            if (n <= 0) { if (hdr_end == std::string::npos || (long long)req.size() < (long long)hdr_end + 4 + clen) refuse = 408; break; }
            req.append(buf, n);
            if (hdr_end == std::string::npos && (hdr_end = req.find("\r\n\r\n")) != std::string::npos) {
                std::string h = req.substr(0, hdr_end);
                for (auto& ch : h) ch = (char)tolower(ch);
                size_t k = h.find("content-length:");
                clen = k == std::string::npos ? 0 : strtoll(h.c_str() + k + 15, nullptr, 10);
                if (clen < 0 || clen > MAX_BODY) { refuse = 413; break; }
            }
            if (hdr_end == std::string::npos && (long long)req.size() > MAX_HEAD) { refuse = 431; break; }
            if (hdr_end != std::string::npos && (long long)req.size() >= (long long)hdr_end + 4 + clen) break;
        }
        if (refuse) {
            if (refuse != 408 || !req.empty())  // nothing at all: just close
                send_all(c, "HTTP/1.1 " + std::to_string(refuse) + (refuse == 413 ? " Payload Too Large" : refuse == 431 ? " Request Header Fields Too Large" : " Request Timeout") +
                            "\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
            if (refuse != 408) fprintf(stderr, "[serve] refused a request: %s\n", refuse == 413 ? "body over 64 MB" : "headers over 1 MB");
            closesocket(c);
            continue;
        }
        std::string method = req.substr(0, req.find(' '));
        size_t ps = req.find(' ') + 1;
        std::string path = req.substr(ps, req.find(' ', ps) - ps);
        path = path.substr(0, path.find('?'));  // e.g. /v1/messages?beta=true
        std::string body = hdr_end == std::string::npos ? "" : req.substr(hdr_end + 4);
        const std::string why = check_caller(hdr_end == std::string::npos ? "" : req.substr(0, hdr_end), loopback, allow_origins, CORS);
        if (!why.empty()) {
            fprintf(stderr, "[serve] refused %s %s: %s\n", method.c_str(), path.c_str(), why.c_str());
            respond(c, 403, "application/json", "{\"error\":{\"message\":\"" + jesc(why) + "\"}}");
            closesocket(c);
            continue;
        }
        if (method == "OPTIONS") { respond(c, 200, "text/plain", ""); }
        else if (method == "GET" && path == "/health") respond(c, 200, "application/json", "{\"status\":\"ok\"}");
        else if (method == "GET" && (path == "/api" || path == "/")) respond(c, 200, "application/json", api_help);
        else if (method == "GET" && path.rfind("/v1/models/", 0) == 0)
            respond(c, 200, "application/json", "{\"id\":\"" + jesc(model_id) + "\",\"object\":\"model\",\"owned_by\":\"moex\",\"type\":\"model\","
                    "\"display_name\":\"" + jesc(model_id) + "\",\"context_length\":" + std::to_string(ss.eng.ctx) + "}");
        else if (method == "GET" && path == "/v1/models")
            respond(c, 200, "application/json", "{\"object\":\"list\",\"data\":[{\"id\":\"" + jesc(model_id) + "\",\"object\":\"model\",\"owned_by\":\"moex\","
                    "\"type\":\"model\",\"display_name\":\"" + jesc(model_id) + "\"}]}");
        else if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions" || path == "/v1/messages" ||
                                      path == "/v1/messages/count_tokens")) {
            JsonParser jp(body); Json r = jp.parse();
            const bool anth = path.rfind("/v1/messages", 0) == 0;
            if (!jp.ok || r.t != Json::OBJ) respond(c, 400, "application/json", anth ? anthropic_error("invalid JSON") : "{\"error\":{\"message\":\"invalid JSON\"}}");
            else if (path == "/v1/chat/completions") openai_serve(c, ss, tk, r, think_default, model_id);
            else if (path == "/v1/completions") completions_serve(c, ss, tk, r, model_id);
            else if (path == "/v1/messages") anthropic_serve(c, ss, tk, r, think_default, model_id);
            else {  // count_tokens
                GenReq q; std::string err;
                if (!anthropic_request(r, think_default, q, err)) respond(c, 400, "application/json", anthropic_error(err));
                else { ss.think = q.think; auto ids = ss.render(q.msgs, q.tools); dump_prompt(ids); respond(c, 200, "application/json", "{\"input_tokens\":" + std::to_string(ids.size()) + "}"); }
            }
        } else respond(c, 404, "application/json", "{\"error\":{\"message\":\"not found\"}}");
        closesocket(c);
    }
}
