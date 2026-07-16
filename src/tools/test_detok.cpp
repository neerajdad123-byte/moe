// MoEx clean-room. Detokenizer correctness check vs known reference decode.
#include <cstdio>
#include <string>
#include <vector>

#include "gguf/gguf.h"
#include "model/tokenizer.h"

using namespace moex;

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1]
                              : "C:/models/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf";
  gguf::Model m;
  std::string err;
  if (!m.open(path, &err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
  Detokenizer tok;
  if (!tok.load(m, &err)) { std::fprintf(stderr, "tok: %s\n", err.c_str()); return 1; }

  std::printf("vocab=%zu bos=%d eos=%d\n", tok.vocab_size(), tok.bos_id(), tok.eos_id());

  struct Case { std::vector<int> ids; const char* expect; };
  Case cases[] = {
    {{785, 6722, 315, 9625, 374}, "The capital of France is"},
    {{9707, 11, 847, 829, 374}, "Hello, my name is"},
    {{750, 75698, 1445, 1648}, "def fibonacci(n):"},
  };
  int fails = 0;
  for (auto& c : cases) {
    std::string got = tok.decode(c.ids);
    bool ok = (got == c.expect);
    std::printf("[%s] \"%s\"%s\n", ok ? "PASS" : "FAIL", got.c_str(),
                ok ? "" : (std::string(" != \"") + c.expect + "\"").c_str());
    if (!ok) ++fails;
  }
  std::printf("%s\n", fails ? "DETOK MISMATCH" : "detok OK");
  return fails ? 1 : 0;
}
