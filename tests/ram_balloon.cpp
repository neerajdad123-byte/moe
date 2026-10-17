// Takes N GB of RAM away from everything else (allocated, touched, locked) for S seconds, so a 24 GB machine behaves
// like a 16 GB one when testing low-RAM mode. No admin rights needed: the locked size fits the working set we ask for.
//   build\ram_balloon.exe 6.5 900
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: ram_balloon GB SECONDS\n"); return 2; }
    const SIZE_T bytes = (SIZE_T)(atof(argv[1]) * 1e9);
    SetProcessWorkingSetSizeEx(GetCurrentProcess(), bytes + (64 << 20), bytes + (128 << 20),
                               QUOTA_LIMITS_HARDWS_MIN_ENABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
    char* p = (char*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) { fprintf(stderr, "cannot allocate %s GB\n", argv[1]); return 1; }
    for (SIZE_T i = 0; i < bytes; i += 4096) p[i] = 1;
    const bool locked = VirtualLock(p, bytes) != 0;
    MEMORYSTATUSEX ms{sizeof ms}; GlobalMemoryStatusEx(&ms);
    printf("balloon %.1f GB %s; RAM free now %.1f GB\n", bytes / 1e9, locked ? "locked" : "NOT locked (may be paged out)", ms.ullAvailPhys / 1e9);
    fflush(stdout);
    Sleep((DWORD)(atof(argv[2]) * 1000));
    return 0;
}
