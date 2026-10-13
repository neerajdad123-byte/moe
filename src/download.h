// `moex download`: fetches the model container from Hugging Face with WinHTTP (part of Windows, no dependencies).
// Resumes interrupted files (Range on a .part file), skips complete ones, checks sizes. Included after serve.h.
#pragma once
#include <winhttp.h>
#include <filesystem>
#pragma comment(lib, "winhttp.lib")

static const char* MODEL_REPO = "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64";  // int4 gs64 container of Qwen3.6-35B-A3B (Apache-2.0)

static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0'); MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}

// GET url (following redirects). Bytes go to `sink`; `from` > 0 asks for a Range. Returns the HTTP status (0 on error).
static int http_get(const std::string& url, uint64_t from, const std::function<bool(const char*, size_t)>& sink, uint64_t* total = nullptr) {
    std::wstring wu = widen(url);
    URL_COMPONENTS uc{sizeof uc}; wchar_t host[256], path[4096];
    uc.lpszHostName = host; uc.dwHostNameLength = 256; uc.lpszUrlPath = path; uc.dwUrlPathLength = 4096;
    if (!WinHttpCrackUrl(wu.c_str(), 0, 0, &uc)) return 0;
    HINTERNET s = WinHttpOpen(L"moex/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    HINTERNET c = s ? WinHttpConnect(s, host, uc.nPort, 0) : nullptr;
    HINTERNET r = c ? WinHttpOpenRequest(c, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0) : nullptr;
    int status = 0;
    if (r) {
        std::wstring hdr = from ? L"Range: bytes=" + std::to_wstring(from) + L"-\r\n" : L"";
        if (WinHttpSendRequest(r, hdr.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : hdr.c_str(), (DWORD)-1, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
            WinHttpReceiveResponse(r, nullptr)) {
            DWORD st = 0, sz = sizeof st;
            WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &st, &sz, WINHTTP_NO_HEADER_INDEX);
            status = (int)st;
            if (total) {
                wchar_t cl[32]; DWORD cls = sizeof cl;
                *total = WinHttpQueryHeaders(r, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_HEADER_NAME_BY_INDEX, cl, &cls, WINHTTP_NO_HEADER_INDEX) ? _wtoi64(cl) : 0;
            }
            std::vector<char> buf(1 << 20);
            DWORD got = 0;
            while (status < 400 && WinHttpReadData(r, buf.data(), (DWORD)buf.size(), &got) && got)
                if (!sink(buf.data(), got)) { status = 0; break; }
        }
    }
    if (r) WinHttpCloseHandle(r);
    if (c) WinHttpCloseHandle(c);
    if (s) WinHttpCloseHandle(s);
    return status;
}

static int download_model(const std::string& dir, const std::string& only) {
    std::string api;
    int st = http_get(std::string("https://huggingface.co/api/models/") + MODEL_REPO + "?blobs=true", 0,
                      [&](const char* p, size_t n) { api.append(p, n); return true; });
    JsonParser jp(api); Json j = jp.parse();
    const Json* sib = j.get("siblings");
    if (st != 200 || !jp.ok || !sib) { fprintf(stderr, "download: cannot list %s (HTTP %d). Check the internet connection.\n", MODEL_REPO, st); return 1; }
    std::vector<std::pair<std::string, uint64_t>> files;
    uint64_t all = 0;
    for (auto& f : sib->a) {
        auto n = f.get("rfilename");
        if (!n || n->t != Json::STR || n->s.rfind(".", 0) == 0) continue;  // skip dotfiles (.gitattributes)
        if (!only.empty() && ("," + only + ",").find("," + n->s + ",") == std::string::npos) continue;
        files.push_back({n->s, (uint64_t)f.num("size", 0)});
        all += files.back().second;
    }
    std::filesystem::create_directories(dir);
    fprintf(stderr, "download: %s -> %s (%zu files, %.1f GB)\n", MODEL_REPO, dir.c_str(), files.size(), all / 1e9);
    uint64_t done_all = 0;
    for (auto& [name, size] : files) {
        std::filesystem::path out = std::filesystem::path(dir) / name, part = out; part += ".part";
        if (std::filesystem::exists(out) && (!size || std::filesystem::file_size(out) == size)) { done_all += size; continue; }
        uint64_t have = std::filesystem::exists(part) ? std::filesystem::file_size(part) : 0;
        FILE* f = _wfopen(part.wstring().c_str(), have ? L"ab" : L"wb");
        if (!f) { fprintf(stderr, "download: cannot write %s\n", part.string().c_str()); return 1; }
        double t0 = now(), tp = 0; uint64_t got = have;
        std::string url = std::string("https://huggingface.co/") + MODEL_REPO + "/resolve/main/" + name;
        int code = http_get(url, have, [&](const char* p, size_t n) {
            if (fwrite(p, 1, n, f) != n) return false;
            got += n;
            if (now() - tp > 1.0) {
                tp = now();
                fprintf(stderr, "\r  %-34s %6.0f / %6.0f MB  %5.1f MB/s  (total %.1f%%)   ", name.c_str(), got / 1e6, size / 1e6,
                        (got - have) / 1e6 / std::max(tp - t0, 1e-3), 100.0 * (done_all + got) / std::max<uint64_t>(all, 1));
            }
            return true;
        });
        fclose(f);
        if ((code != 200 && code != 206) || (size && std::filesystem::file_size(part) != size)) {
            fprintf(stderr, "\ndownload: %s failed (HTTP %d, %llu of %llu bytes). Run `moex download` again to resume.\n", name.c_str(), code,
                    (unsigned long long)std::filesystem::file_size(part), (unsigned long long)size);
            return 1;
        }
        std::filesystem::rename(part, out);
        done_all += size;
        fprintf(stderr, "\r  %-34s done %*s\n", name.c_str(), 50, "");
    }
    fprintf(stderr, "download: complete. Run: moex chat\n");
    return 0;
}

// Where the model lives: --model, else MOEX_MODEL, else models\ next to moex.exe (falling back to the dev path).
static std::string exe_dir() {
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    return std::filesystem::path(exe).parent_path().string();
}
static std::string default_model_dir() {
    if (const char* e = getenv("MOEX_MODEL")) return e;
    if (const char* e = getenv("QWEN36_MODEL")) return e;
    std::filesystem::path p = std::filesystem::path(exe_dir()) / "models" / "qwen36-35b-a3b-i4-gs64";
    if (!std::filesystem::exists(p / "config.json") && std::filesystem::exists("C:\\models\\qwen36_i4_gs64\\config.json"))
        return "C:\\models\\qwen36_i4_gs64";
    return p.string();
}

// Free RAM and VRAM, printed around loading. Afterwards both look nearly full by design: free VRAM became expert
// slots and the experts are locked in RAM (so CPU misses never wait on the disk).
static void mem_report(const char* when) {
    MEMORYSTATUSEX ms{sizeof ms}; GlobalMemoryStatusEx(&ms);
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    fprintf(stderr, "[%s] RAM %.1f GB free of %.1f GB | VRAM %.1f GB free of %.1f GB%s\n", when,
            ms.ullAvailPhys / 1e9, ms.ullTotalPhys / 1e9, fr / 1e9, tot / 1e9,
            strcmp(when, "after load") ? "" : "  (by design: free VRAM became expert slots, RAM holds the experts)");
}
