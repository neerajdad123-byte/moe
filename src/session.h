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
    std::map<std::string, std::vector<int>> replies;  // our own replies, so a resent history reuses their exact ids
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
                add("assistant\n<think>\n\n</think>\n\n");
                auto it = replies.find(m.content);
                if (it != replies.end()) ids.insert(ids.end(), it->second.begin(), it->second.end());
                else add(m.content);
            } else add(m.role + "\n" + m.content);
            ids.push_back(im_end);
            add("\n");
        }
        ids.push_back(im_start);
        add(think ? "assistant\n" : "assistant\n<think>\n\n</think>\n\n");
        return ids;
    }

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
        for (int i = 0; i < max_new && next != im_end && next != eot; i++) {
            out.push_back(next);
            if (!on_tok(next) || (int)ctx.size() + 1 >= eng.ctx) break;
            int pos = (int)ctx.size();
            ctx.push_back(next);
            next = eng.step(next, pos, true);
        }
        return out;
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
