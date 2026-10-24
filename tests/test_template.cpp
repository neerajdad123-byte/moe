// Chat-template check without the model (no GPU): renders one request with Session::render, moex's Qwen3.6 template
// plus the variants it picks from a chat_template.jinja, and prints the prompt's token ids. tools/check_template.py
// --offline compares them with the Jinja template itself. build.bat -> build\test_template.exe
//   build\test_template.exe TOKENIZER.json TEMPLATE_DIR REQUEST.json   (TEMPLATE_DIR/chat_template.jinja, if any)
#include <cstdint>
#include <deque>
#include <vector>
struct Engine {  // render() needs none of it; generate() is compiled but never called here
    int ctx = 1 << 30;
    bool has_mtp = false;
    int prefill(const int*, int, int) { return 0; }
    void reset() {}
    int step(int, int, bool) { return 0; }
    std::vector<int> mtp_round(int, int) { return {}; }
};
#include "../src/session.h"

int main(int argc, char** argv) {
    if (argc != 4) { fprintf(stderr, "usage: test_template TOKENIZER.json TEMPLATE_DIR REQUEST.json\n"); return 2; }
    Tokenizer tk(argv[1]);
    Engine e;
    Session ss(e, tk);
    ss.load_template(argv[2]);
    FILE* f = fopen(argv[3], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    std::string s;
    char b[4096];
    for (size_t n; (n = fread(b, 1, sizeof b, f)) > 0;) s.append(b, n);
    fclose(f);
    JsonParser jp(s);
    const Json r = jp.parse();
    std::vector<ChatMsg> msgs;
    for (auto& x : r.get("messages")->a) {  // as serve.h's openai_request
        ChatMsg cm;
        cm.role = x.str("role", "user");
        if (cm.role == "developer") cm.role = "system";
        cm.content = x.str("content");
        if (auto rc = x.get("reasoning_content")) if (rc->t == Json::STR) { cm.reasoning = rc->s; cm.has_reasoning = true; }
        if (auto tcs = x.get("tool_calls"))
            for (auto& t : tcs->a) {
                ToolCall tc; tc.id = t.str("id"); tc.args = Json::object();
                const Json* fn = t.get("function");
                tc.name = fn->str("name");
                if (auto a = fn->get("arguments")) {
                    if (a->t == Json::OBJ) tc.args = *a;
                    else if (a->t == Json::STR) { JsonParser ap(a->s); Json v = ap.parse(); if (ap.done() && v.t == Json::OBJ) tc.args = v; }
                }
                cm.tool_calls.push_back(tc);
            }
        msgs.push_back(cm);
    }
    std::vector<Json> tools;
    if (auto t = r.get("tools")) for (auto& x : t->a) tools.push_back(x);
    ss.think = r.boolean("think", false);
    const std::vector<int> ids = ss.render(msgs, tools);
    for (size_t i = 0; i < ids.size(); i++) printf(i ? ",%d" : "%d", ids[i]);
    printf("\n");
    return 0;
}
