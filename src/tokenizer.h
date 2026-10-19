// Qwen3.6 tokenizer from the model's own tokenizer.json: byte-level BPE with the Qwen pre-tokenizer regex,
// implemented without regex/ICU (hand-written matcher + generated Unicode class tables). Verified token-for-token
// against HF `tokenizers` (tools/check_tokenizer.py).
// shortcut: no NFC normalization (input is assumed already NFC, as typed/pasted text almost always is); add NFC
// tables if a mismatch ever shows up.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include "unicode_tab.h"

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;      // byte-level token string -> id
    std::vector<std::string> id2tok;                 // id -> byte-level token string (added tokens: their content)
    std::unordered_map<std::string, int> merge_rank; // "a b" -> rank
    std::vector<std::pair<std::string, int>> added;  // added tokens, longest first
    std::vector<char> is_special;                    // id -> special (skipped when decoding for display)
    uint32_t byte2cp[256];
    std::unordered_map<uint32_t, uint8_t> cp2byte;
    std::unordered_map<std::string, std::vector<int>> cache;

    explicit Tokenizer(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path.c_str()); exit(1); }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        std::string j(n, '\0'); if (fread(j.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "tokenizer: read failed\n"); exit(1); }
        fclose(f);
        size_t m = j.find("\"model\""), v = j.find("\"vocab\"", m), mg = j.find("\"merges\"", m);
        if (m == std::string::npos || v == std::string::npos || mg == std::string::npos) { fprintf(stderr, "tokenizer: unexpected tokenizer.json\n"); exit(1); }
        // vocab: { "tok": id, ... }
        size_t p = j.find('{', v) + 1;
        while (true) {
            skip_ws(j, p);
            if (j[p] == '}') break;
            std::string k = jstr(j, p); skip_ws(j, p); p++;  // ':'
            skip_ws(j, p); int id = (int)strtol(j.c_str() + p, nullptr, 10);
            while (j[p] != ',' && j[p] != '}') p++;
            if (j[p] == ',') p++;
            vocab[k] = id;
            if ((int)id2tok.size() <= id) id2tok.resize(id + 1);
            id2tok[id] = k;
        }
        // merges: [ "a b", ... ]
        p = j.find('[', mg) + 1;
        for (int r = 0;; r++) {
            skip_ws(j, p);
            if (j[p] == ']') break;
            merge_rank[jstr(j, p)] = r;
            skip_ws(j, p); if (j[p] == ',') p++;
        }
        // added tokens: [{"id": N, "content": "...", ..., "special": bool}, ...]
        size_t a = j.find("\"added_tokens\""), ae = j.find(']', j.find('[', a));
        for (size_t q = j.find('{', a); q != std::string::npos && q < ae; q = j.find('{', q + 1)) {
            size_t ki = j.find("\"id\"", q), kc = j.find("\"content\"", q), ks = j.find("\"special\"", q);
            int id = (int)strtol(j.c_str() + j.find(':', ki) + 1, nullptr, 10);
            size_t cp = j.find('"', j.find(':', kc) + 1); std::string c = jstr(j, cp);
            bool sp = j.compare(j.find_first_not_of(" :", ks + 9), 4, "true") == 0;
            added.push_back({c, id});
            if ((int)id2tok.size() <= id) id2tok.resize(id + 1);
            id2tok[id] = c;
            if ((int)is_special.size() <= id) is_special.resize(id + 1, 0);
            is_special[id] = sp;
        }
        is_special.resize(id2tok.size(), 0);
        std::sort(added.begin(), added.end(), [](auto& x, auto& y) { return x.first.size() > y.first.size(); });
        // GPT-2 bytes_to_unicode
        int k = 0;
        for (int b = 0; b < 256; b++) {
            bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            byte2cp[b] = keep ? b : 256 + k++;
            cp2byte[byte2cp[b]] = (uint8_t)b;
        }
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> out;
        size_t i = 0, start = 0;
        while (i <= text.size()) {  // added tokens are matched whole, before pre-tokenization
            int hit = -1; size_t len = 0;
            if (i < text.size() && text[i] == '<')
                for (auto& [s, id] : added) if (text.compare(i, s.size(), s) == 0) { hit = id; len = s.size(); break; }
            if (hit >= 0 || i == text.size()) {
                encode_plain(text.substr(start, i - start), out);
                if (hit >= 0) out.push_back(hit);
                i += hit >= 0 ? len : 1; start = i;
            } else i++;
        }
        return out;
    }

    // Raw bytes of one token (added tokens: their text).
    std::string piece(int id) const {
        if (id < 0 || id >= (int)id2tok.size()) return "";
        const std::string& s = id2tok[id];
        if (id < (int)is_special.size() && id >= (int)vocab.size()) return s;  // added token
        std::string b;
        for (size_t p = 0; p < s.size();) { uint32_t cp = next_cp(s, p); auto it = cp2byte.find(cp); if (it != cp2byte.end()) b += (char)it->second; }
        return b;
    }
    bool special(int id) const { return id >= 0 && id < (int)is_special.size() && is_special[id]; }
    int id_of(const std::string& s) const {
        for (auto& [c, id] : added) if (c == s) return id;
        auto it = vocab.find(s); return it == vocab.end() ? -1 : it->second;
    }

private:
    static void skip_ws(const std::string& j, size_t& p) { while (p < j.size() && (j[p] == ' ' || j[p] == '\n' || j[p] == '\r' || j[p] == '\t')) p++; }
    static void put_utf8(std::string& o, uint32_t c) {
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xC0 | c >> 6); o += (char)(0x80 | (c & 63)); }
        else if (c < 0x10000) { o += (char)(0xE0 | c >> 12); o += (char)(0x80 | (c >> 6 & 63)); o += (char)(0x80 | (c & 63)); }
        else { o += (char)(0xF0 | c >> 18); o += (char)(0x80 | (c >> 12 & 63)); o += (char)(0x80 | (c >> 6 & 63)); o += (char)(0x80 | (c & 63)); }
    }
    // JSON string at j[p] == '"', with escapes (\uXXXX incl. surrogate pairs); p ends after the closing quote.
    static std::string jstr(const std::string& j, size_t& p) {
        std::string o; p++;
        while (j[p] != '"') {
            char c = j[p++];
            if (c != '\\') { o += c; continue; }
            char e = j[p++];
            switch (e) {
                case 'n': o += '\n'; break; case 't': o += '\t'; break; case 'r': o += '\r'; break;
                case 'b': o += '\b'; break; case 'f': o += '\f'; break;
                case 'u': {
                    uint32_t cp = (uint32_t)strtoul(j.substr(p, 4).c_str(), nullptr, 16); p += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && j[p] == '\\' && j[p + 1] == 'u') {
                        uint32_t lo = (uint32_t)strtoul(j.substr(p + 2, 4).c_str(), nullptr, 16); p += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(o, cp); break;
                }
                default: o += e;  // \" \\ \/
            }
        }
        p++;
        return o;
    }
    static uint32_t next_cp(const std::string& s, size_t& p) {
        unsigned char c = s[p];
        int n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        uint32_t cp = n == 1 ? c : n == 2 ? c & 31 : n == 3 ? c & 15 : c & 7;
        for (int k = 1; k < n && p + k < s.size(); k++) cp = cp << 6 | (s[p + k] & 63);
        p += n;
        return cp;
    }
    static bool in(const uint32_t (*t)[2], size_t n, uint32_t c) {
        size_t lo = 0, hi = n;
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (c < t[mid][0]) hi = mid; else if (c > t[mid][1]) lo = mid + 1; else return true; }
        return false;
    }
    static bool isL(uint32_t c) { return in(UNI_L, sizeof UNI_L / sizeof UNI_L[0], c); }
    static bool isM(uint32_t c) { return in(UNI_M, sizeof UNI_M / sizeof UNI_M[0], c); }
    static bool isN(uint32_t c) { return in(UNI_N, sizeof UNI_N / sizeof UNI_N[0], c); }
    static bool isS(uint32_t c) { return in(UNI_S, sizeof UNI_S / sizeof UNI_S[0], c); }
    static bool isNL(uint32_t c) { return c == '\r' || c == '\n'; }

    // The Qwen pre-tokenizer, alternatives in regex order (first match wins):
    //  (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
    //  | \s*[\r\n]+ | \s+(?!\S) | \s+
    void encode_plain(const std::string& s, std::vector<int>& out) {
        std::vector<uint32_t> c; std::vector<size_t> at;  // codepoints and their byte offsets
        for (size_t p = 0; p < s.size();) { at.push_back(p); c.push_back(next_cp(s, p)); }
        at.push_back(s.size());
        const size_t n = c.size();
        for (size_t i = 0; i < n;) {
            size_t e = match(c, i);
            bpe(s.substr(at[i], at[e] - at[i]), out);
            i = e;
        }
    }
    size_t match(const std::vector<uint32_t>& c, size_t i) const {
        const size_t n = c.size();
        auto lower = [](uint32_t x) { return x >= 'A' && x <= 'Z' ? x + 32 : x; };
        if (c[i] == '\'' && i + 1 < n) {
            uint32_t a = lower(c[i + 1]), b = i + 2 < n ? lower(c[i + 2]) : 0;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        }
        auto LM = [&](size_t k) { return k < n && (isL(c[k]) || isM(c[k])); };
        {   // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            size_t k = i;
            if (!isNL(c[k]) && !isL(c[k]) && !isN(c[k]) && LM(k + 1)) k++;
            if (LM(k)) { while (LM(k)) k++; return k; }
        }
        if (isN(c[i])) return i + 1;
        {   // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
            auto P = [&](size_t k) { return k < n && !isS(c[k]) && !isL(c[k]) && !isM(c[k]) && !isN(c[k]); };
            size_t k = i + (c[i] == ' ' && P(i + 1) ? 1 : 0);
            if (P(k)) { while (P(k)) k++; while (k < n && isNL(c[k])) k++; return k; }
        }
        size_t r = i; while (r < n && isS(c[r])) r++;  // the whitespace run [i, r)
        if (r > i) {
            for (size_t k = r; k > i; k--) if (isNL(c[k - 1])) return k;  // \s*[\r\n]+ ends after the run's last newline
            if (r == n) return r;                                          // \s+(?!\S) at the end of the text
            if (r - i >= 2) return r - 1;                                  // ... else leave one space for the next word
            return r;                                                      // \s+
        }
        return i + 1;  // unreachable for valid input; never stall
    }
    void bpe(const std::string& word, std::vector<int>& out) {
        auto ci = cache.find(word);
        if (ci != cache.end()) { out.insert(out.end(), ci->second.begin(), ci->second.end()); return; }
        // Lowest-rank merge first, leftmost among equal ranks (as HF), with a heap over the adjacent pairs and a linked
        // list of symbols: O(n log n). Rescanning every pair after each merge was quadratic: a 50,000-letter word
        // (a DNA string, a blob) took 29 s.
        struct Sym { std::string s; int prev, next; };
        std::vector<Sym> sy;
        for (unsigned char b : word) { std::string u; put_utf8(u, byte2cp[b]); sy.push_back({u, (int)sy.size() - 1, (int)sy.size() + 1}); }
        if (!sy.empty()) sy.back().next = -1;
        struct Cand { int rank, l, r; size_t ll, lr; };  // a pair as it was when queued (lengths detect later merges)
        auto later = [](const Cand& a, const Cand& b) { return a.rank != b.rank ? a.rank > b.rank : a.l > b.l; };
        std::priority_queue<Cand, std::vector<Cand>, decltype(later)> q(later);
        auto queue_pair = [&](int l, int r) {
            if (l < 0 || r < 0) return;
            auto it = merge_rank.find(sy[l].s + " " + sy[r].s);
            if (it != merge_rank.end()) q.push({it->second, l, r, sy[l].s.size(), sy[r].s.size()});
        };
        for (int k = 0; k + 1 < (int)sy.size(); k++) queue_pair(k, k + 1);
        while (!q.empty()) {
            const Cand c = q.top(); q.pop();
            if (sy[c.l].next != c.r || sy[c.l].s.size() != c.ll || sy[c.r].s.size() != c.lr) continue;  // stale
            sy[c.l].s += sy[c.r].s;
            sy[c.l].next = sy[c.r].next;
            if (sy[c.r].next >= 0) sy[sy[c.r].next].prev = c.l;
            sy[c.r].next = -2;  // merged away
            queue_pair(sy[c.l].prev, c.l);
            queue_pair(c.l, sy[c.l].next);
        }
        std::vector<int> ids;
        for (int k = sy.empty() ? -1 : 0; k >= 0; k = sy[k].next) { auto it = vocab.find(sy[k].s); if (it != vocab.end()) ids.push_back(it->second); }
        if (cache.size() < 200000) cache[word] = ids;
        out.insert(out.end(), ids.begin(), ids.end());
    }
};
