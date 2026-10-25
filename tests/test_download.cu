// `moex download` helpers (src/download.h): file names from the Hugging Face listing must stay inside the model
// folder, the Linux curl command quotes the URL, and resuming a .part file gives the right file whether or not the
// server honors Range. Uses a tiny HTTP server on 127.0.0.1 in this process; no internet needed.
//   build\test_download.exe
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include "../src/plat.h"  // GlobalMemoryStatusEx for mem_report
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SD_SEND SHUT_WR
#define closesocket close
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#include "../src/json.h"
#include "../src/download.h"

// Serves BODY. honor_range: answer "Range: bytes=N-" with 206 and the rest; otherwise always 200 and the whole body.
static std::string BODY;
static std::atomic<bool> honor_range{true};
static std::atomic<int> requests{0};
static void http_server(SOCKET ls) {
    for (;;) {
        SOCKET c = accept(ls, nullptr, nullptr);
        if (c == INVALID_SOCKET) return;
        std::string req; char buf[4096];
        while (req.find("\r\n\r\n") == std::string::npos) { int n = recv(c, buf, sizeof buf, 0); if (n <= 0) break; req.append(buf, n); }
        requests++;
        size_t k = req.find("Range: bytes="), from = k == std::string::npos ? 0 : strtoull(req.c_str() + k + 13, nullptr, 10);
        std::string resp;
        if (from && honor_range && from >= BODY.size())
            resp = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        else if (from && honor_range)
            resp = "HTTP/1.1 206 Partial Content\r\nContent-Length: " + std::to_string(BODY.size() - from) + "\r\nConnection: close\r\n\r\n" + BODY.substr(from);
        else resp = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(BODY.size()) + "\r\nConnection: close\r\n\r\n" + BODY;
        send(c, resp.data(), (int)resp.size(), 0);
        shutdown(c, SD_SEND); closesocket(c);
    }
}
static std::string slurp(const std::filesystem::path& p) { std::ifstream f(p, std::ios::binary); return {std::istreambuf_iterator<char>(f), {}}; }
static void put(const std::filesystem::path& p, const std::string& s) { std::ofstream(p, std::ios::binary) << s; }

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

    // resuming a .part file
#ifdef _WIN32
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
    int alen = sizeof(sockaddr_in);
#else
    socklen_t alen = sizeof(sockaddr_in);
#endif
    SOCKET ls = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 4) || getsockname(ls, (sockaddr*)&a, &alen)) { printf("cannot listen\n"); return 1; }
    std::thread(http_server, ls).detach();
    const std::string url = "http://127.0.0.1:" + std::to_string(ntohs(a.sin_port)) + "/f.bin";
    for (int i = 0; i < 100000; i++) BODY += (char)('a' + i * 7 % 26);
    const fs::path tmp = fs::temp_directory_path() / "moex_test_download";
    fs::create_directories(tmp);
    const fs::path f = tmp / "f.bin", part = tmp / "f.bin.part";
    auto run = [&](const std::string& what, bool honor, const std::string& part_before) {
        fs::remove(f); fs::remove(part);
        if (!part_before.empty()) put(part, part_before);
        honor_range = honor;
        uint64_t done = 0;
        bool ok = fetch_file(url, f, BODY.size(), "f.bin", done, BODY.size());
        check(what, ok && fs::exists(f) && !fs::exists(part) && slurp(f) == BODY);
    };
    run("fresh download", true, "");
    run("resume, server honors Range (206)", true, BODY.substr(0, 40000));
    run("resume, server ignores Range (200): starts over", false, BODY.substr(0, 40000));
    run(".part larger than the file: starts over", true, BODY + "junk");
    run(".part already complete: renamed", true, BODY);
    fs::remove_all(tmp);
    printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
