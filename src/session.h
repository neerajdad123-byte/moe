// A conversation on top of the Engine: the engine state (KV cache, DeltaNet state) is kept between turns, so a new
// turn only prefills its new tokens. Included by moex.cu after Engine.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "tokenizer.h"

struct ChatMsg { std::string role, content; };

struct Session {
    Engine& eng;
    Tokenizer& tk;
    bool think = false;           // let the model think (<think> ... </think>) instead of an empty think block
    std::vector<int> ctx;         // tokens whose effect is in the engine state, at positions 0..ctx.size()-1
    // Our own replies by their text (what a client sends back), so a resent history reuses their exact ids. head is
    // what came after "assistant\n" in the prompt: the empty think block, or nothing when the model thought (its
    // thinking is then in ids, so the history still matches the engine state and only the new turn is prefilled).
    struct Reply { std::string head; std::vector<int> ids; };
    std::map<std::string, Reply> replies;
    int im_start, im_end, eot;

    Session(Engine& e, Tokenizer& t) : eng(e), tk(t) {
        im_start = tk.id_of("<|im_start|>"); im_end = tk.id_of("<|im_end|>"); eot = tk.id_of("<|endoftext|>");
    }

    // ChatML. Our earlier replies are inserted as the ids we generated (re-tokenizing their text can differ), so a
    // client that resends the whole history extends ctx and only the new turn is prefilled.
    std::vector<int> render(const std::vector<ChatMsg>& msgs) {
        std::vector<int> ids;
        auto add = [&](const std::string& s) { auto v = tk.encode(s); ids.insert(ids.end(), v.begin(), v.end()); };
        for (auto& m : msgs) {
            ids.push_back(im_start);
            if (m.role == "assistant") {
                auto it = replies.find(m.content);
                if (it != replies.end()) {  // tokenized exactly as the generation prompt was
                    add("assistant\n" + it->second.head);
                    ids.insert(ids.end(), it->second.ids.begin(), it->second.ids.end());
                } else { add("assistant\n" + EMPTY_THINK); add(m.content); }
            } else add(m.role + "\n" + m.content);
            ids.push_back(im_end);
            add("\n");
        }
        ids.push_back(im_start);
        add("assistant\n" + gen_head());
        return ids;
    }
    static inline const std::string EMPTY_THINK = "<think>\n\n</think>\n\n";
    std::string gen_head() const { return think ? "" : EMPTY_THINK; }
    // After generate(): `text` is the reply as the client will send it back in the history.
    void remember(const std::string& text, const std::vector<int>& out) { replies[text] = {gen_head(), out}; }

    // Runs `prompt` and streams up to max_new generated tokens to on_tok (return false to stop early). Stops at
    // <|im_end|> / <|endoftext|> (not emitted, not fed). Returns the generated ids.
    std::vector<int> generate(const std::vector<int>& prompt, int max_new, const std::function<bool(int)>& on_tok) {
        size_t common = 0;
        while (common < ctx.size() && common < prompt.size() && ctx[common] == prompt[common]) common++;
        if ((int)prompt.size() >= eng.ctx) { fprintf(stderr, "[session] prompt (%zu tokens) exceeds --ctx %d\n", prompt.size(), eng.ctx); return {}; }
        int next;
        if (common == ctx.size() && common < prompt.size()) {
            next = eng.prefill(prompt.data() + common, (int)(prompt.size() - common), (int)common);
        } else {  // the history changed: start over (the DeltaNet state cannot be rewound)
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
                if (ts.size() == 2) ctx.push_back(ts[0]);
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
