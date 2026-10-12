// Minimal safetensors reader for the colibri container (one file per layer + globals).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

struct StTensor {
    std::string file, dtype;
    std::vector<int64_t> shape;
    uint64_t off = 0, bytes = 0;  // absolute file offset
};

struct Container {
    std::string root;
    std::unordered_map<std::string, StTensor> t;

    explicit Container(const std::string& dir) : root(dir) {
        for (auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".safetensors") index(e.path().string());
        if (t.empty()) { fprintf(stderr, "no safetensors in %s\n", dir.c_str()); exit(1); }
    }

    const StTensor& at(const std::string& name) const {
        auto it = t.find(name);
        if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
        return it->second;
    }

    void read_into(const std::string& name, void* dst) const {
        const StTensor& s = at(name);
        FILE* f = fopen(s.file.c_str(), "rb");
        if (!f || _fseeki64(f, (int64_t)s.off, SEEK_SET) || fread(dst, 1, s.bytes, f) != s.bytes) {
            fprintf(stderr, "read failed: %s\n", name.c_str()); exit(1);
        }
        fclose(f);
    }

    std::vector<uint8_t> read(const std::string& name) const {
        std::vector<uint8_t> v(at(name).bytes);
        read_into(name, v.data());
        return v;
    }

    // Pointer into a read-only memory map of the tensor's file (the OS page cache is the RAM tier).
    const uint8_t* map(const std::string& name) {
        const StTensor& s = at(name);
        auto it = maps.find(s.file);
        if (it == maps.end()) {
            HANDLE f = CreateFileA(s.file.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
            HANDLE m = f == INVALID_HANDLE_VALUE ? nullptr : CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
            const void* p = m ? MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
            if (!p) { fprintf(stderr, "mmap failed: %s\n", s.file.c_str()); exit(1); }
            it = maps.emplace(s.file, (const uint8_t*)p).first;  // mapped for the process lifetime
        }
        return it->second + s.off;
    }

private:
    std::unordered_map<std::string, const uint8_t*> maps;

    // The header is flat JSON: {"name":{"dtype":"F16","shape":[..],"data_offsets":[a,b]},...}
    void index(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        uint64_t n = 0;
        if (!f || fread(&n, 8, 1, f) != 1) { fprintf(stderr, "bad file %s\n", path.c_str()); exit(1); }
        std::string h(n, '\0');
        if (fread(h.data(), 1, n, f) != n) { fprintf(stderr, "bad header %s\n", path.c_str()); exit(1); }
        fclose(f);
        size_t p = 0;
        auto str = [&](size_t& q) { size_t a = h.find('"', q) + 1, b = h.find('"', a); q = b + 1; return h.substr(a, b - a); };
        auto nums = [&](size_t q) {
            std::vector<int64_t> v;
            size_t a = h.find('[', q) + 1, b = h.find(']', a);
            for (char* s = h.data() + a; s < h.data() + b;) { v.push_back(strtoll(s, &s, 10)); while (s < h.data() + b && (*s == ',' || *s == ' ')) s++; }
            return v;
        };
        p = h.find('{') + 1;
        while (true) {
            size_t q = h.find('"', p);
            if (q == std::string::npos) break;
            std::string key = str(q);
            size_t obj = h.find('{', q), end = h.find('}', obj);
            p = end + 1;
            if (key == "__metadata__") continue;
            std::string body = h.substr(obj, end - obj);
            StTensor s;
            s.file = path;
            size_t d = body.find("\"dtype\""); d = body.find(':', d); s.dtype = [&] { size_t a = body.find('"', d) + 1; return body.substr(a, body.find('"', a) - a); }();
            size_t base = obj;
            s.shape = nums(base + body.find("\"shape\""));
            auto offs = nums(base + body.find("\"data_offsets\""));
            s.off = 8 + n + offs[0];
            s.bytes = offs[1] - offs[0];
            t[key] = s;
        }
    }
};
