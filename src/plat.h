// Platform layer. Windows: <windows.h>. Linux: the few Win32 calls moex uses, on POSIX (mmap, /proc/meminfo,
// O_DIRECT reads on a small thread pool for the overlapped ones), so the engine keeps one code path.
#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

typedef unsigned long DWORD;  // printed with %lu, as on Windows
typedef size_t SIZE_T;
typedef int BOOL;
typedef intptr_t HANDLE;  // a file descriptor
#define TRUE 1
#define FALSE 0
#define INVALID_HANDLE_VALUE ((HANDLE)-1)
#define GENERIC_READ 0
#define FILE_SHARE_READ 0
#define OPEN_EXISTING 0
#define FILE_FLAG_NO_BUFFERING 1
#define FILE_FLAG_OVERLAPPED 0
#define FILE_FLAG_SEQUENTIAL_SCAN 0
#define FILE_FLAG_RANDOM_ACCESS 0
#define MEM_COMMIT 0
#define MEM_RESERVE 0
#define PAGE_READWRITE 0
#define QUOTA_LIMITS_HARDWS_MIN_DISABLE 0
#define QUOTA_LIMITS_HARDWS_MAX_DISABLE 0
#define ERROR_IO_PENDING 997
#define _fseeki64 fseeko
#define _ftelli64 ftello

static thread_local DWORD plat_err = 0;
inline DWORD GetLastError() { return plat_err; }

struct MEMORYSTATUSEX { DWORD dwLength; uint64_t ullTotalPhys, ullAvailPhys; };
inline BOOL GlobalMemoryStatusEx(MEMORYSTATUSEX* m) {
    m->ullTotalPhys = m->ullAvailPhys = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return FALSE;
    char k[64]; unsigned long long v;
    while (fscanf(f, "%63s %llu kB", k, &v) == 2) {
        if (!strcmp(k, "MemTotal:")) m->ullTotalPhys = v << 10;
        if (!strcmp(k, "MemAvailable:")) m->ullAvailPhys = v << 10;
    }
    fclose(f);
    return TRUE;
}

inline int GetCurrentProcess() { return 0; }
// Linux keeps file pages cached without a working-set quota; mlock needs RLIMIT_MEMLOCK (often 8 MB): best effort.
inline BOOL SetProcessWorkingSetSize(int, SIZE_T, SIZE_T) { return TRUE; }
inline BOOL SetProcessWorkingSetSizeEx(int, SIZE_T, SIZE_T, DWORD) { return TRUE; }
inline BOOL VirtualLock(void* p, SIZE_T n) { return mlock(p, n) == 0; }
inline void* VirtualAlloc(void*, SIZE_T n, DWORD, DWORD) {
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
struct WIN32_MEMORY_RANGE_ENTRY { void* VirtualAddress; SIZE_T NumberOfBytes; };
inline BOOL PrefetchVirtualMemory(int, SIZE_T n, WIN32_MEMORY_RANGE_ENTRY* r, DWORD) {
    const uintptr_t pg = 4096;
    for (SIZE_T i = 0; i < n; i++) {
        uintptr_t a = (uintptr_t)r[i].VirtualAddress & ~(pg - 1);
        madvise((void*)a, (uintptr_t)r[i].VirtualAddress + r[i].NumberOfBytes - a, MADV_WILLNEED);
    }
    return TRUE;
}

inline HANDLE CreateFileA(const char* path, DWORD, DWORD, void*, DWORD, DWORD flags, void*) {
    int fd = open(path, O_RDONLY | ((flags & FILE_FLAG_NO_BUFFERING) ? O_DIRECT : 0));
    if (fd < 0 && (flags & FILE_FLAG_NO_BUFFERING)) fd = open(path, O_RDONLY);  // a filesystem without O_DIRECT
    return fd < 0 ? INVALID_HANDLE_VALUE : (HANDLE)fd;
}
inline BOOL CloseHandle(HANDLE h) { return close((int)h) == 0; }
inline HANDLE CreateEventA(void*, BOOL, BOOL, void*) { return 0; }  // overlapped reads below complete by flag
inline BOOL ResetEvent(HANDLE) { return TRUE; }

// Overlapped reads: pread on a pool of threads (glibc's POSIX AIO runs one file's requests one after another).
struct OVERLAPPED {
    uintptr_t Internal, InternalHigh; DWORD Offset, OffsetHigh; HANDLE hEvent;
    int state = 0;  // 1 queued, 2 done (atomic builtins: the struct stays copyable, it lives in vectors)
    long long got = 0; int err = 0;
    HANDLE fd; void* buf; DWORD n;
};
struct PlatIo {
    std::mutex mu; std::condition_variable cv; std::deque<OVERLAPPED*> q;
    PlatIo() { for (int i = 0; i < 8; i++) std::thread([this] { run(); }).detach(); }
    void run() {
        while (true) {
            OVERLAPPED* o;
            { std::unique_lock<std::mutex> g(mu); cv.wait(g, [&] { return !q.empty(); }); o = q.front(); q.pop_front(); }
            const off_t off = (off_t)o->Offset | (off_t)o->OffsetHigh << 32;
            long long r = 0;
            while (r < (long long)o->n) {
                const size_t want = o->n - r;
                ssize_t k = pread((int)o->fd, (char*)o->buf + r, want, off + r);
                if (k < 0) { if (errno == EINTR) continue; o->err = errno; break; }
                r += k;
                // short = end of file: the 4 KB-rounded read of a file's last expert (Windows returns it short too);
                // reading on from an unaligned offset would fail under O_DIRECT
                if ((size_t)k < want) break;
            }
            o->got = r;
            __atomic_store_n(&o->state, 2, __ATOMIC_RELEASE);
        }
    }
    // never destroyed: its detached threads wait on cv until the process ends, and glibc's condition variable
    // destructor (run at exit for a static) waits for its waiters: moex hung on exit after any low-RAM run
    static PlatIo& get() { static PlatIo* p = new PlatIo; return *p; }
};
inline BOOL ReadFile(HANDLE h, void* buf, DWORD n, DWORD* got, OVERLAPPED* o) {
    if (!o) {
        ssize_t k = read((int)h, buf, n);
        if (k < 0) { plat_err = errno; return FALSE; }
        if (got) *got = (DWORD)k;
        return TRUE;
    }
    o->fd = h; o->buf = buf; o->n = n; o->err = 0; __atomic_store_n(&o->state, 1, __ATOMIC_RELAXED);
    PlatIo& p = PlatIo::get();
    { std::lock_guard<std::mutex> g(p.mu); p.q.push_back(o); }
    p.cv.notify_one();
    plat_err = ERROR_IO_PENDING;
    return FALSE;
}
inline bool HasOverlappedIoCompleted(OVERLAPPED* o) { return __atomic_load_n(&o->state, __ATOMIC_ACQUIRE) == 2; }
inline BOOL GetOverlappedResult(HANDLE, OVERLAPPED* o, DWORD* got, BOOL wait) {
    while (__atomic_load_n(&o->state, __ATOMIC_ACQUIRE) != 2) {
        if (!wait) { plat_err = ERROR_IO_PENDING; return FALSE; }
        std::this_thread::yield();
    }
    if (o->err) { plat_err = o->err; return FALSE; }
    if (got) *got = (DWORD)o->got;
    return TRUE;
}
#endif
