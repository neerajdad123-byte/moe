// Tokenizes every file in a directory with src/tokenizer.h and prints "name: id id id ..." (and a decode round trip
// check), for tools/check_tokenizer.py to compare against HF tokenizers.
//   build\test_tokenizer.exe <tokenizer.json> <dir>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "../src/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_tokenizer tokenizer.json dir\n"); return 2; }
    Tokenizer tk(argv[1]);
    for (auto& e : std::filesystem::directory_iterator(argv[2])) {
        std::ifstream f(e.path(), std::ios::binary); std::stringstream ss; ss << f.rdbuf();
        std::string text = ss.str();
        auto ids = tk.encode(text);
        std::string back; for (int id : ids) back += tk.piece(id);
        printf("%s:%s", e.path().filename().string().c_str(), back == text ? "" : " ROUNDTRIP_FAIL");
        for (int id : ids) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
