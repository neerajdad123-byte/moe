// `moex download` helpers (src/download.h): file names from the Hugging Face listing must stay inside the model
// folder. No network needed.
//   build\test_download.exe
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include <windows.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#include "../src/json.h"
#include "../src/download.h"

int main() {
    int fails = 0;
    auto check = [&](const std::string& what, bool ok) { printf("  %-62s %s\n", what.c_str(), ok ? "PASS" : "FAIL"); fails += !ok; };
    namespace fs = std::filesystem;
    const fs::path dir = "C:\\models\\m";
    fs::path out;
    for (const char* ok : {"config.json", "model-00001-of-00004.safetensors", "sub/tokenizer.json", "a/./b.json", "a..b.json"})
        check(std::string("accepted: ") + ok, model_file_path(dir, ok, out) && out == (dir / ok).lexically_normal());
    for (const char* bad : {"", ".", "..", "../evil.dll", "..\\evil.dll", "a/../../evil.dll", "a/..", "/etc/passwd", "\\evil.dll",
                            "C:\\Windows\\evil.dll", "C:evil.dll", "\\\\server\\share\\x", "config.json:stream", "a/b/../../../x"})
        check(std::string("refused:  ") + bad, !model_file_path(dir, bad, out));
    // Linux hands the URL to curl through the shell: a ' in a file name must not end the quoted word
    check("shell_quote: plain URL", shell_quote("https://h/a b.json") == "'https://h/a b.json'");
    check("shell_quote: ' becomes '\\''", shell_quote("x';rm -rf ~;'") == "'x'\\'';rm -rf ~;'\\'''");
    check("shell_quote: $() and backquotes stay literal", shell_quote("$(id)`id`") == "'$(id)`id`'");
    check("shell_quote: empty", shell_quote("") == "''");
    printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
