// read_user_env (src/download.h), which `moex install` uses to read the user PATH before adding its folder: a value
// longer than the old fixed 32,768-character buffer must be read in full, and "not set" must differ from an error.
// Writes and deletes only a throwaway variable (MOEX_TEST_USERENV) under HKCU\Environment; PATH is not touched.
//   build\test_userenv.exe
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
    const wchar_t* name = L"MOEX_TEST_USERENV";
    int fails = 0;
    auto check = [&](const char* what, bool ok) { printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL"); fails += !ok; };
    std::wstring v;
    RegDeleteKeyValueW(HKEY_CURRENT_USER, L"Environment", name);
    check("not set: read succeeds, empty", read_user_env(name, v) && v.empty());
    std::wstring big;
    for (int i = 0; big.size() < 40000; i++) big += L"C:\\some\\folder\\number_" + std::to_wstring(i) + L";";
    check("set a 40,000-character value", set_user_env(name, big));
    check("read it back in full (old code: 32,768-char buffer failed -> \"\")", read_user_env(name, v) && v == big);
    check("short value", set_user_env(name, L"x;y") && read_user_env(name, v) && v == L"x;y");
    RegDeleteKeyValueW(HKEY_CURRENT_USER, L"Environment", name);
    check("deleted again: not set", read_user_env(name, v) && v.empty());
    printf(fails ? "%d FAILED\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
