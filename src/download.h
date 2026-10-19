// `moex download`: fetches the model container from Hugging Face with WinHTTP (part of Windows, no dependencies;
// Linux: curl). Resumes interrupted files (Range on a .part file), skips complete ones, checks sizes. Included after serve.h.
#pragma once
#include <filesystem>

static const char* MODEL_REPO = "Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64";  // int4 gs64 container of Qwen3.6-35B-A3B (Apache-2.0)

// s as one POSIX shell word: in single quotes, each ' written as '\'' (Linux passes URLs to curl through popen)
static std::string shell_quote(const std::string& s) {
    std::string q = "'";
    for (char ch : s) q += ch == '\'' ? std::string("'\\''") : std::string(1, ch);
    return q + "'";
}

#ifdef _WIN32
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

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
#else
// shortcut: curl's exit code stands in for the HTTP status (22 = HTTP error); `total` stays unset (no caller uses it)
static int http_get(const std::string& url, uint64_t from, const std::function<bool(const char*, size_t)>& sink, uint64_t* = nullptr) {
    std::string cmd = "curl -sSLf" + (from ? " -r " + std::to_string(from) + "-" : std::string()) + " " + shell_quote(url);
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return 0;
    std::vector<char> buf(1 << 20);
    bool ok = true;
    for (size_t got; ok && (got = fread(buf.data(), 1, buf.size(), p)) > 0;) ok = sink(buf.data(), got);
    const int rc = pclose(p);
    return ok && rc == 0 ? (from ? 206 : 200) : 0;
}
#endif

// One file, resumable through a .part file. size 0 = unknown (no size check). false on failure (message printed).
static bool fetch_file(const std::string& url, const std::filesystem::path& out, uint64_t size, const std::string& name,
                       uint64_t& done_all, uint64_t all) {
    std::filesystem::path part = out; part += ".part";
    if (std::filesystem::exists(out) && (!size || std::filesystem::file_size(out) == size)) { done_all += size; return true; }
    uint64_t have = std::filesystem::exists(part) ? std::filesystem::file_size(part) : 0;
#ifdef _WIN32
    FILE* f = _wfopen(part.wstring().c_str(), have ? L"ab" : L"wb");
#else
    FILE* f = fopen(part.c_str(), have ? "ab" : "wb");
#endif
    if (!f) { fprintf(stderr, "download: cannot write %s\n", part.string().c_str()); return false; }
    double t0 = now(), tp = 0; uint64_t got = have;
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
        return false;
    }
    std::filesystem::rename(part, out);
    done_all += size;
    fprintf(stderr, "\r  %-34s done %*s\n", name.c_str(), 50, "");
    return true;
}

// The MTP draft head (multi-token prediction) of Qwen3.6-35B-A3B, converted to the same int4 format by
// tools/fetch_mtp.py from the official Apache-2.0 weights, hosted as a release asset of this project.
static const char* MTP_URL = "https://github.com/neerajdad123-byte/moe/releases/download/weights-mtp-1/mtp.safetensors";
static const uint64_t MTP_BYTES = 531715224;

// A file name from the Hugging Face listing, as a path inside the model folder `dir`. false for any name that could
// write elsewhere: absolute, with a drive ("C:x") or a ':' (alternate stream), with a ".." part, or empty.
static bool model_file_path(const std::filesystem::path& dir, const std::string& name, std::filesystem::path& out) {
    namespace fs = std::filesystem;
    if (name.empty() || name.find(':') != std::string::npos) return false;
    const fs::path rel(name);
    if (rel.is_absolute() || rel.has_root_name() || rel.has_root_directory()) return false;
    for (auto& part : rel) if (part == "..") return false;
    const fs::path norm = rel.lexically_normal();
    if (norm.empty() || norm == "." || *norm.begin() == "..") return false;
    out = (dir / norm).lexically_normal();
    return true;
}

static int download_model(const std::string& dir, const std::string& only) {
    std::string api;
    int st = http_get(std::string("https://huggingface.co/api/models/") + MODEL_REPO + "?blobs=true", 0,
                      [&](const char* p, size_t n) { api.append(p, n); return true; });
    JsonParser jp(api); Json j = jp.parse();
    const Json* sib = j.get("siblings");
    if (st != 200 || !jp.ok || !sib) { fprintf(stderr, "download: cannot list %s (HTTP %d). Check the internet connection.\n", MODEL_REPO, st); return 1; }
    auto wanted = [&](const std::string& n) { return only.empty() || ("," + only + ",").find("," + n + ",") != std::string::npos; };
    struct File { std::string name; std::filesystem::path out; uint64_t size; };
    std::vector<File> files;
    uint64_t all = 0;
    for (auto& f : sib->a) {
        auto n = f.get("rfilename");
        if (!n || n->t != Json::STR || n->s.rfind(".", 0) == 0 || !wanted(n->s)) continue;  // skip dotfiles (.gitattributes)
        std::filesystem::path out;
        if (!model_file_path(dir, n->s, out)) {
            fprintf(stderr, "download: the listing of %s has a file name outside the model folder (%s); nothing downloaded.\n", MODEL_REPO, n->s.c_str());
            return 1;
        }
        files.push_back({n->s, out, (uint64_t)f.num("size", 0)});
        all += files.back().size;
    }
    const bool mtp = wanted("mtp.safetensors");
    if (mtp) all += MTP_BYTES;
    std::filesystem::create_directories(dir);
    fprintf(stderr, "download: %s%s -> %s (%zu files, %.1f GB)\n", MODEL_REPO, mtp ? " + MTP head" : "", dir.c_str(), files.size() + mtp, all / 1e9);
    uint64_t done_all = 0;
    for (auto& [name, out, size] : files)
        if (!fetch_file(std::string("https://huggingface.co/") + MODEL_REPO + "/resolve/main/" + name, out, size, name, done_all, all))
            return 1;
    if (mtp && !fetch_file(MTP_URL, std::filesystem::path(dir) / "mtp.safetensors", MTP_BYTES, "mtp.safetensors", done_all, all)) return 1;
    fprintf(stderr, "download: complete. Run: moex chat --load\n");
    return 0;
}

// Where the model lives: --model, else MOEX_MODEL, else models\ next to moex.exe (falling back to the dev path).
static std::string exe_dir() {
#ifdef _WIN32
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    return std::filesystem::path(exe).parent_path().string();
#else
    return std::filesystem::read_symlink("/proc/self/exe").parent_path().string();
#endif
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

// `moex install`: copy moex.exe (+ moex_hot.bin, README.md, LICENSE) to %LOCALAPPDATA%\Programs\moex and add that
// folder to the user PATH, so `moex` works in any new terminal. A model already downloaded next to the copy being
// installed stays where it is: MOEX_MODEL is set to it. No admin rights needed (per-user environment).
#ifndef _WIN32
// Linux: a link in ~/.local/bin (on PATH in most distributions); the model stays next to the real binary.
static int install_moex() {
    namespace fs = std::filesystem;
    const char* home = getenv("HOME");
    if (!home) { fprintf(stderr, "install: HOME is not set\n"); return 1; }
    const fs::path dst = fs::path(home) / ".local" / "bin" / "moex";
    std::error_code ec;
    fs::create_directories(dst.parent_path(), ec);
    fs::remove(dst, ec);
    fs::create_symlink(fs::read_symlink("/proc/self/exe"), dst, ec);
    if (ec) { fprintf(stderr, "install: cannot create %s (%s)\n", dst.c_str(), ec.message().c_str()); return 1; }
    fprintf(stderr, "install: %s -> this moex. If `moex` is not found in a new terminal, add ~/.local/bin to PATH.\n", dst.c_str());
    return 0;
}
#else
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "user32.lib")
static bool set_user_env(const wchar_t* name, const std::wstring& value) {
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Environment", 0, KEY_SET_VALUE, &k) != ERROR_SUCCESS) return false;
    bool ok = RegSetValueExW(k, name, 0, REG_EXPAND_SZ, (const BYTE*)value.c_str(), (DWORD)((value.size() + 1) * 2)) == ERROR_SUCCESS;
    RegCloseKey(k);
    return ok;
}
// A user environment variable from the registry, any length. false = it could not be read (not "not set", which
// is an empty value): the caller must then not write it back, or a failed read would wipe the user's PATH.
static bool read_user_env(const wchar_t* name, std::wstring& out) {
    out.clear();
    const DWORD flags = RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND;
    for (int attempt = 0; attempt < 3; attempt++) {  // the value may grow between the size query and the read
        DWORD sz = 0;
        LSTATUS r = RegGetValueW(HKEY_CURRENT_USER, L"Environment", name, flags, nullptr, nullptr, &sz);
        if (r == ERROR_FILE_NOT_FOUND) return true;
        if (r != ERROR_SUCCESS) return false;
        std::vector<wchar_t> buf(sz / sizeof(wchar_t) + 2, 0);
        sz = (DWORD)(buf.size() * sizeof(wchar_t));
        r = RegGetValueW(HKEY_CURRENT_USER, L"Environment", name, flags, nullptr, buf.data(), &sz);
        if (r == ERROR_SUCCESS) { out = buf.data(); return true; }
        if (r != ERROR_MORE_DATA) return false;
    }
    return false;
}
static std::wstring get_user_env(const wchar_t* name) { std::wstring v; read_user_env(name, v); return v; }
static int install_moex() {
    namespace fs = std::filesystem;
    wchar_t la[MAX_PATH];
    if (!GetEnvironmentVariableW(L"LOCALAPPDATA", la, MAX_PATH)) { fprintf(stderr, "install: LOCALAPPDATA is not set\n"); return 1; }
    const fs::path src = exe_dir(), dst = fs::path(la) / "Programs" / "moex";
    std::error_code ec;
    fs::create_directories(dst, ec);
    if (!fs::equivalent(src, dst, ec)) {
        for (const char* f : {"moex.exe", "moex_hot.bin", "README.md", "LICENSE"})
            if (fs::exists(src / f) && !fs::copy_file(src / f, dst / f, fs::copy_options::overwrite_existing, ec)) {
                fprintf(stderr, "install: cannot copy %s to %s (%s). Is moex running from there? Close it and retry.\n", f, dst.string().c_str(), ec.message().c_str());
                return 1;
            }
        const fs::path model = src / "models" / "qwen36-35b-a3b-i4-gs64";
        if (fs::exists(model / "config.json") && get_user_env(L"MOEX_MODEL").empty() && set_user_env(L"MOEX_MODEL", model.wstring()))
            fprintf(stderr, "install: the model stays in %s (MOEX_MODEL)\n", model.string().c_str());
    }
    std::wstring path, d = dst.wstring();
    if (!read_user_env(L"Path", path)) {
        fprintf(stderr, "install: moex is in %s, but your PATH could not be read, so it was left unchanged.\n"
                        "Add that folder to PATH by hand (Windows search: \"Edit environment variables for your account\").\n", dst.string().c_str());
        return 1;
    }
    bool present = false;
    for (size_t a = 0; a <= path.size();) {
        size_t b = path.find(L';', a); if (b == std::wstring::npos) b = path.size();
        if (_wcsicmp(path.substr(a, b - a).c_str(), d.c_str()) == 0) present = true;
        a = b + 1;
    }
    if (!present && !set_user_env(L"Path", path.empty() || path.back() == L';' ? path + d : path + L";" + d)) {
        fprintf(stderr, "install: cannot update PATH\n"); return 1;
    }
    DWORD_PTR r;  // tell Explorer and new terminals that the environment changed
    SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM)L"Environment", SMTO_ABORTIFHUNG, 2000, &r);
    fprintf(stderr, "install: moex is in %s and on your PATH.\nOpen a NEW terminal, then run from anywhere:\n  moex download\n  moex chat --load\n",
            dst.string().c_str());
    return 0;
}
#endif
