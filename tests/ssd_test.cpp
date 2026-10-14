// Cold read latency of one expert (1.77 MB) from the SSD, bypassing the file cache, split into P pieces issued
// together with overlapped I/O (file handles opened once, as the engine would). Decides whether demand reads
// should be split.   build\ssd_test.exe C:\models\qwen36_i4_gs64
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "../src/st.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    Container C(argc > 1 ? argv[1] : "C:\\models\\qwen36_i4_gs64");
    struct Range { HANDLE h; uint64_t a, b; };
    std::map<std::string, HANDLE> handles;
    auto handle = [&](const std::string& f) {
        auto it = handles.find(f);
        if (it != handles.end()) return it->second;
        HANDLE h = CreateFileA(f.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "open %s failed\n", f.c_str()); exit(1); }
        return handles[f] = h;
    };
    std::vector<std::vector<Range>> experts;  // per expert: 4 KB-aligned ranges of merged_weight and qs
    for (int l = 0; l < 40; l++)
        for (int e = 0; e < 256; e++) {
            std::vector<Range> r;
            for (const char* t : {"merged_weight", "qs"}) {
                const StTensor& s = C.at("model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + t);
                r.push_back({handle(s.file), s.off & ~4095ull, (s.off + s.bytes + 4095) & ~4095ull});
            }
            experts.push_back(r);
        }
    uint8_t* buf = (uint8_t*)VirtualAlloc(nullptr, 4 << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::vector<OVERLAPPED> ov(64);
    std::vector<HANDLE> ev(64);
    for (auto& x : ev) x = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    std::mt19937 rng(7);

    for (int P : {1, 2, 4, 8, 16}) {
        std::vector<double> lat;
        for (int i = 0; i < 150; i++) {
            const auto& ex = experts[rng() % experts.size()];
            // pieces: each range cut into ~P/2 parts (the two tensors together give ~P reads in flight)
            struct Piece { HANDLE h; uint64_t off, len, dst; };
            std::vector<Piece> pcs;
            uint64_t dst = 0;
            for (const Range& r : ex) {
                const uint64_t n = r.b - r.a, parts = std::max<uint64_t>(1, P * n / 1769472 + (P > 1 ? 0 : 0));
                const uint64_t step = std::max<uint64_t>(4096, (n / parts + 4095) & ~4095ull);
                for (uint64_t o = 0; o < n; o += step) { pcs.push_back({r.h, r.a + o, std::min(step, n - o), dst + o}); }
                dst += n;
            }
            double t0 = now();
            for (size_t k = 0; k < pcs.size(); k++) {
                ov[k] = {};
                ov[k].Offset = (DWORD)pcs[k].off; ov[k].OffsetHigh = (DWORD)(pcs[k].off >> 32); ov[k].hEvent = ev[k];
                ResetEvent(ev[k]);
                if (!ReadFile(pcs[k].h, buf + pcs[k].dst, (DWORD)pcs[k].len, nullptr, &ov[k]) && GetLastError() != ERROR_IO_PENDING) {
                    fprintf(stderr, "read failed %lu\n", GetLastError()); return 1;
                }
            }
            for (size_t k = 0; k < pcs.size(); k++) {
                DWORD got = 0;
                if (!GetOverlappedResult(pcs[k].h, &ov[k], &got, TRUE) && GetLastError() != ERROR_HANDLE_EOF) {
                    fprintf(stderr, "read error %lu\n", GetLastError()); return 1;
                }  // a short read only happens at the end of a file (4 KB rounding)
            }
            lat.push_back(now() - t0);
        }
        std::sort(lat.begin(), lat.end());
        printf("one expert in %2d pieces: median %.2f ms, p90 %.2f ms\n", P, 1e3 * lat[lat.size() / 2], 1e3 * lat[lat.size() * 9 / 10]);
    }
    return 0;
}
