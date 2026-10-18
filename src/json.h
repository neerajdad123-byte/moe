// Minimal JSON: parse, build, and dump. Objects keep their key order and numbers keep their text, so dump()
// writes a value back as Python's json.dumps(ensure_ascii=False) does (the chat template's tojson).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static std::string jesc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"') o += "\\\""; else if (c == '\\') o += "\\\\"; else if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t"; else if (c == '\b') o += "\\b"; else if (c == '\f') o += "\\f";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c;
    }
    return o;
}

struct Json {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double n = 0;
    std::string s;  // STR: the string; NUM: its text as written
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;
    const Json* get(const std::string& k) const {
        if (t != OBJ) return nullptr;
        for (auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    double num(const std::string& k, double def) const { auto v = get(k); return v && v->t == NUM ? v->n : def; }
    bool boolean(const std::string& k, bool def) const { auto v = get(k); return v && v->t == BOOL ? v->b : def; }
    std::string str(const std::string& k, const std::string& def = "") const { auto v = get(k); return v && v->t == STR ? v->s : def; }
    // builders
    static Json string(const std::string& v) { Json j; j.t = STR; j.s = v; return j; }
    static Json object() { Json j; j.t = OBJ; return j; }
    Json& set(const std::string& k, Json v) {
        for (auto& kv : o) if (kv.first == k) { kv.second = std::move(v); return *this; }
        o.push_back({k, std::move(v)}); return *this;
    }
    std::string dump() const {
        switch (t) {
            case NUL: return "null";
            case BOOL: return b ? "true" : "false";
            case NUM: return s;
            case STR: return "\"" + jesc(s) + "\"";
            case ARR: { std::string r = "["; for (size_t i = 0; i < a.size(); i++) r += (i ? ", " : "") + a[i].dump(); return r + "]"; }
            case OBJ: { std::string r = "{"; for (size_t i = 0; i < o.size(); i++) r += (i ? ", \"" : "\"") + jesc(o[i].first) + "\": " + o[i].second.dump(); return r + "}"; }
        }
        return "null";
    }
};

struct JsonParser {
    const std::string& j; size_t p = 0; bool ok = true;
    explicit JsonParser(const std::string& s) : j(s) {}
    void ws() { while (p < j.size() && strchr(" \t\r\n", j[p])) p++; }
    bool done() { ws(); return ok && p == j.size(); }  // the whole text was one value
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
                p++; v.set(k, parse()); ws();
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
        else {
            char* e; v.t = Json::NUM; v.n = strtod(j.c_str() + p, &e);
            if (e == j.c_str() + p) ok = false;
            v.s = j.substr(p, e - (j.c_str() + p)); p = e - j.c_str();
        }
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
        if (p >= j.size()) ok = false;  // unterminated string
        p++;
        return o;
    }
};
