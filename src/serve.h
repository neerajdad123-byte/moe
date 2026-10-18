// OpenAI-compatible HTTP server (Winsock, no dependencies): GET /v1/models, GET /health, GET /api (help), POST /v1/chat/completions
// (streaming SSE or a single JSON response). One request at a time; listens on 127.0.0.1 unless told otherwise.
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
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <vector>
#pragma comment(lib, "ws2_32.lib")

// ---- minimal JSON (parse + escape) ----
struct Json {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double n = 0; std::string s;
    std::vector<Json> a; std::map<std::string, Json> o;
    const Json* get(const std::string& k) const { auto it = o.find(k); return t == OBJ && it != o.end() ? &it->second : nullptr; }
    double num(const std::string& k, double def) const { auto v = get(k); return v && v->t == NUM ? v->n : def; }
    bool boolean(const std::string& k, bool def) const { auto v = get(k); return v && v->t == BOOL ? v->b : def; }
};
struct JsonParser {
    const std::string& j; size_t p = 0; bool ok = true;
    explicit JsonParser(const std::string& s) : j(s) {}
    void ws() { while (p < j.size() && strchr(" \t\r\n", j[p])) p++; }
    Json parse() {
        ws(); Json v;
        if (p >= j.size()) { ok = false; return v; }
        char c = j[p];
        if (c == '{') {
            v.t = Json::OBJ; p++; ws();
            if (p < j.size() && j[p] == '}') { p++; return v; }
            while (ok) {
                ws(); if (p >= j.size() || j[p] != '"') { ok = false; break; }
                std::string k = str(); ws();
                if (p >= j.size() || j[p] != ':') { ok = false; break; }
                p++; v.o[k] = parse(); ws();
                if (p < j.size() && j[p] == ',') { p++; continue; }
                if (p < j.size() && j[p] == '}') { p++; break; }
                ok = false;
            }
        } else if (c == '[') {
            v.t = Json::ARR; p++; ws();
            if (p < j.size() && j[p] == ']') { p++; return v; }
            while (ok) {
                v.a.push_back(parse()); ws();
                if (p < j.size() && j[p] == ',') { p++; continue; }
                if (p < j.size() && j[p] == ']') { p++; break; }
                ok = false;
            }
        } else if (c == '"') { v.t = Json::STR; v.s = str(); }
        else if (!j.compare(p, 4, "true")) { v.t = Json::BOOL; v.b = true; p += 4; }
        else if (!j.compare(p, 5, "false")) { v.t = Json::BOOL; p += 5; }
        else if (!j.compare(p, 4, "null")) { p += 4; }
        else { char* e; v.t = Json::NUM; v.n = strtod(j.c_str() + p, &e); if (e == j.c_str() + p) ok = false; p = e - j.c_str(); }
        return v;
    }
    std::string str() {
        std::string o; p++;
        while (p < j.size() && j[p] != '"') {
            char c = j[p++];
            if (c != '\\') { o += c; continue; }
            if (p >= j.size()) break;
            char e = j[p++];
            if (e == 'n') o += '\n'; else if (e == 't') o += '\t'; else if (e == 'r') o += '\r';
            else if (e == 'b') o += '\b'; else if (e == 'f') o += '\f';
            else if (e == 'u' && p + 4 <= j.size()) {
                uint32_t cp = (uint32_t)strtoul(j.substr(p, 4).c_str(), nullptr, 16); p += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && p + 6 <= j.size() && j[p] == '\\' && j[p + 1] == 'u') {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + ((uint32_t)strtoul(j.substr(p + 2, 4).c_str(), nullptr, 16) - 0xDC00); p += 6;
                }
                if (cp < 0x80) o += (char)cp;
                else if (cp < 0x800) { o += (char)(0xC0 | cp >> 6); o += (char)(0x80 | (cp & 63)); }
                else if (cp < 0x10000) { o += (char)(0xE0 | cp >> 12); o += (char)(0x80 | (cp >> 6 & 63)); o += (char)(0x80 | (cp & 63)); }
                else { o += (char)(0xF0 | cp >> 18); o += (char)(0x80 | (cp >> 12 & 63)); o += (char)(0x80 | (cp >> 6 & 63)); o += (char)(0x80 | (cp & 63)); }
            } else o += e;
        }
        p++;
        return o;
    }
};
static std::string jesc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t"; else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
    }
    return o;
}

// ---- HTTP ----
static bool send_all(SOCKET c, const std::string& s) {
    for (size_t p = 0; p < s.size();) { int n = send(c, s.data() + p, (int)(s.size() - p), 0); if (n <= 0) return false; p += n; }
    return true;
}
static const char* CORS = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
static void respond(SOCKET c, int code, const std::string& type, const std::string& body) {
    const char* msg = code == 200 ? "OK" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Error";
    send_all(c, "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + type + "\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n" + CORS + "Connection: close\r\n\r\n" + body);
}

static void serve(Session& ss, Tokenizer& tk, const std::string& host, int port, const std::string& model_id) {
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN);  // a client that hangs up mid-stream must not end the server
#endif
    const bool think_default = ss.think;  // `moex serve --think`
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    int yes = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof yes);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 16)) { fprintf(stderr, "[serve] cannot listen on %s:%d (port in use? choose another with --port)\n", host.c_str(), port); return; }
    const std::string base = "http://" + std::string(host == "0.0.0.0" ? "127.0.0.1" : host) + ":" + std::to_string(port);
    const std::string ps_example = "Invoke-RestMethod " + base + "/v1/chat/completions -Method Post -ContentType application/json "
                                   "-Body '{\"model\":\"" + model_id + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'";
    const std::string py_example = "from openai import OpenAI; c = OpenAI(base_url=\"" + base + "/v1\", api_key=\"none\"); "
                                   "print(c.chat.completions.create(model=\"" + model_id + "\", messages=[{\"role\": \"user\", \"content\": \"Hello!\"}]).choices[0].message.content)";
    // GET /api (and /): what this server offers and how to call it
    const std::string api_help = "{\"server\":\"moex\",\"model\":\"" + jesc(model_id) + "\",\"base_url\":\"" + jesc(base) + "/v1\",\"api_key\":\"any\","
        "\"endpoints\":{\"POST /v1/chat/completions\":\"OpenAI chat completions; set \\\"stream\\\": true for streaming. Also: temperature, top_p, top_k, "
        "seed, stop, max_tokens; thinking: chat_template_kwargs.enable_thinking or reasoning_effort (default: serve --think), sent as reasoning_content, the answer as content\",\"GET /v1/models\":\"the model id\",\"GET /health\":\"{\\\"status\\\":\\\"ok\\\"}\","
        "\"GET /api\":\"this page\"},\"examples\":{\"powershell\":\"" + jesc(ps_example) + "\",\"python\":\"" + jesc(py_example) + "\"}}";
    fprintf(stderr,
            "\n[serve] moex is serving %s - OpenAI-compatible API\n"
            "  Base URL:   %s/v1   (model \"%s\", any API key)\n"
            "  Endpoints:  POST /v1/chat/completions   GET /v1/models   GET /health   GET /api (help)\n"
            "  Try it in PowerShell:\n    %s\n"
            "  Python (pip install openai):\n    %s\n"
            "  Open WebUI, editors and other apps: add an OpenAI-compatible connection with the base URL above.\n"
            "  %s\n"
            "  Ctrl+C stops the server.\n\n",
            model_id.c_str(), base.c_str(), model_id.c_str(), ps_example.c_str(), py_example.c_str(),
            host == "127.0.0.1" ? "Only this PC can connect (--host 0.0.0.0 opens it to your network)." : "Other devices on your network can connect too.");
    while (true) {
        SOCKET c = accept(ls, nullptr, nullptr);
        if (c == INVALID_SOCKET) continue;
        std::string req; char buf[65536];
        size_t hdr_end = std::string::npos; long clen = 0;
        while (true) {  // headers, then Content-Length bytes of body
            int n = recv(c, buf, sizeof buf, 0);
            if (n <= 0) break;
            req.append(buf, n);
            if (hdr_end == std::string::npos && (hdr_end = req.find("\r\n\r\n")) != std::string::npos) {
                std::string h = req.substr(0, hdr_end);
                for (auto& ch : h) ch = (char)tolower(ch);
                size_t k = h.find("content-length:");
                clen = k == std::string::npos ? 0 : strtol(h.c_str() + k + 15, nullptr, 10);
            }
            if (hdr_end != std::string::npos && req.size() >= hdr_end + 4 + clen) break;
            if (req.size() > (64u << 20)) break;  // refuse absurd requests
        }
        std::string method = req.substr(0, req.find(' '));
        size_t ps = req.find(' ') + 1;
        std::string path = req.substr(ps, req.find(' ', ps) - ps);
        std::string body = hdr_end == std::string::npos ? "" : req.substr(hdr_end + 4);
        if (method == "OPTIONS") { respond(c, 200, "text/plain", ""); }
        else if (method == "GET" && path == "/health") respond(c, 200, "application/json", "{\"status\":\"ok\"}");
        else if (method == "GET" && (path == "/api" || path == "/")) respond(c, 200, "application/json", api_help);
        else if (method == "GET" && path == "/v1/models")
            respond(c, 200, "application/json", "{\"object\":\"list\",\"data\":[{\"id\":\"" + jesc(model_id) + "\",\"object\":\"model\",\"owned_by\":\"moex\"}]}");
        else if (method == "POST" && path == "/v1/chat/completions") {
            JsonParser jp(body); Json r = jp.parse();
            const Json* msgs = r.get("messages");
            if (!jp.ok || !msgs || msgs->t != Json::ARR || msgs->a.empty()) { respond(c, 400, "application/json", "{\"error\":{\"message\":\"messages missing or invalid JSON\"}}"); closesocket(c); continue; }
            std::vector<ChatMsg> m;
            for (auto& x : msgs->a) {
                ChatMsg cm; auto ro = x.get("role"), co = x.get("content");
                cm.role = ro && ro->t == Json::STR ? ro->s : "user";
                if (co && co->t == Json::STR) cm.content = co->s;
                else if (co && co->t == Json::ARR)  // content parts: keep the text ones
                    for (auto& part : co->a) { auto tx = part.get("text"); if (tx && tx->t == Json::STR) cm.content += tx->s; }
                m.push_back(cm);
            }
            const bool stream = r.boolean("stream", false);
            int max_new = (int)r.num("max_completion_tokens", r.num("max_tokens", 2048));
            // thinking: chat_template_kwargs.enable_thinking, else reasoning_effort ("none" = off), else `serve --think`
            auto kw = r.get("chat_template_kwargs");
            auto re = r.get("reasoning_effort");
            ss.think = kw && kw->get("enable_thinking") ? kw->boolean("enable_thinking", think_default)
                     : re && re->t == Json::STR ? re->s != "none"
                     : think_default;
            Engine::Sampling sp;
            sp.temp = (float)r.num("temperature", ss.think ? 1.0 : 0.7);
            sp.top_p = (float)r.num("top_p", ss.think ? 0.95 : 0.8);
            sp.top_k = (int)r.num("top_k", 20);
            sp.rep = (float)r.num("repetition_penalty", 1.0);
            sp.seed = (uint64_t)r.num("seed", 0);
            if (sp.seed) ss.eng.rng.seed(sp.seed);
            ss.eng.samp = sp;
            std::vector<std::string> stops;
            if (auto st = r.get("stop")) { if (st->t == Json::STR) stops.push_back(st->s); else for (auto& x : st->a) if (x.t == Json::STR) stops.push_back(x.s); }
            stops.erase(std::remove(stops.begin(), stops.end(), std::string()), stops.end());
            const std::string id = "chatcmpl-" + std::to_string((long long)time(nullptr)) + std::to_string(rand());
            const long long created = (long long)time(nullptr);
            auto prompt = ss.render(m);
            if (stream) send_all(c, std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n") + CORS + "Connection: close\r\n\r\n");
            auto chunk = [&](const std::string& delta, const char* finish, const char* field = "content") {
                std::string d = finish ? "{}" : "{\"" + std::string(field) + "\":\"" + jesc(delta) + "\"}";
                return "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created) +
                       ",\"model\":\"" + jesc(model_id) + "\",\"choices\":[{\"index\":0,\"delta\":" + d + ",\"finish_reason\":" +
                       (finish ? "\"" + std::string(finish) + "\"" : "null") + "}]}\n\n";
            };
            // the thinking goes out as reasoning_content (as llama.cpp, vLLM and DeepSeek send it: Open WebUI and other
            // clients show it as a thinking section), the answer as content; stop strings apply to the answer only
            Utf8Stream us; ThinkSplit split(ss.think); std::string text, reasoning; bool alive = true, hit_stop = false; size_t sent = 0;
            auto feed = [&](const std::string& piece, bool last) {
                std::string rd, ad;
                split.push(piece, rd, ad, last);
                if (!rd.empty()) { reasoning += rd; if (stream && alive) alive = send_all(c, chunk(rd, nullptr, "reasoning_content")); }
                text += ad;
                for (auto& s : stops) { size_t k = text.find(s); if (k != std::string::npos) { text.resize(k); hit_stop = true; } }
                // hold back a tail that could still become a stop string
                size_t hold = 0; for (auto& s : stops) hold = std::max(hold, s.size() - 1);
                size_t can = hit_stop || last ? text.size() : (text.size() > hold ? text.size() - hold : 0);
                if (stream && alive && can > sent) { alive = send_all(c, chunk(text.substr(sent, can - sent), nullptr)); sent = can; }
            };
            const double t0 = now();
            auto out = ss.generate(prompt, max_new, [&](int t) {
                feed(us.push(tk.piece(t)), false);
                return alive && !hit_stop;
            });
            if (!hit_stop) feed(us.held, true);
            ss.remember(text, out);
            ss.eng.save_hot();
            fprintf(stderr, "[serve] %zu prompt tokens, %zu generated, %.1f s, %.1f tok/s overall\n", prompt.size(), out.size(), now() - t0,
                    out.size() / std::max(now() - t0, 1e-9));
            const char* finish = hit_stop || (int)out.size() < max_new ? "stop" : "length";
            if (stream) {
                if (alive && text.size() > sent) send_all(c, chunk(text.substr(sent), nullptr));
                send_all(c, chunk("", finish));
                send_all(c, "data: [DONE]\n\n");
            } else {
                respond(c, 200, "application/json",
                        "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) + ",\"model\":\"" +
                        jesc(model_id) + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + jesc(text) + "\"" +
                        (reasoning.empty() ? "" : ",\"reasoning_content\":\"" + jesc(reasoning) + "\"") +
                        "},\"finish_reason\":\"" + finish + "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt.size()) +
                        ",\"completion_tokens\":" + std::to_string(out.size()) + ",\"total_tokens\":" + std::to_string(prompt.size() + out.size()) + "}}");
            }
        } else respond(c, 404, "application/json", "{\"error\":{\"message\":\"not found\"}}");
        closesocket(c);
    }
}
