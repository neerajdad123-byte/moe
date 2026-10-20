// `moex bench`: speed on the kind of prompt most people type (a short everyday question, a few paragraphs back), in
// ~2-4 minutes with one model load: 3 warm-up prompts (expert cache, file cache, graphs: not counted), then the same
// 3 questions with MTP off, MTP on, and MTP on + --fast; each result prints as it finishes. Each question is a new
// conversation with chat's sampling and a fixed seed. None are in data/prompts.txt (the prompts behind the shipped
// moex_hot.bin). Included by moex.cu after session.h.
#pragma once

static const char* BENCH_PROMPTS[] = {
    "Why do we get hiccups, and what actually helps stop them?",
    "How does a microwave oven heat food?",
    "How do noise-cancelling headphones work?",
};
constexpr int BENCH_TOKENS = 128;  // reply cap: a couple of paragraphs

static std::string cpu_name() {
    int r[4]; char s[49] = {};
#ifdef _WIN32
    for (int i = 0; i < 3; i++) { __cpuid(r, 0x80000002 + i); memcpy(s + 16 * i, r, 16); }
#else
    for (int i = 0; i < 3; i++) { __cpuid(0x80000002 + i, r[0], r[1], r[2], r[3]); memcpy(s + 16 * i, r, 16); }
#endif
    std::string n = s;
    n.erase(0, n.find_first_not_of(' '));
    return n;
}

static std::string driver_version() {  // from nvidia-smi (installed with every NVIDIA driver); "?" without it
#ifdef _WIN32
    FILE* p = _popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>nul", "r");
#else
    FILE* p = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null", "r");
#endif
    char b[64] = {};
    if (p) {
        if (!fgets(b, sizeof b, p)) b[0] = 0;
#ifdef _WIN32
        _pclose(p);
#else
        pclose(p);
#endif
    }
    std::string v = b;
    while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ')) v.pop_back();
    return v.empty() ? "?" : v;
}

static int run_bench(Engine& eng, Tokenizer& tk, const std::string& model) {
    Session ss(eng, tk);
    const int np = (int)(sizeof BENCH_PROMPTS / sizeof *BENCH_PROMPTS);
    const bool mtp = eng.has_mtp;  // loaded (MTP head downloaded, enough RAM): the MTP phases switch it per prompt
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, 0));
    MEMORYSTATUSEX ms{sizeof ms}; GlobalMemoryStatusEx(&ms);
    const std::string spec = std::string(p.name) + " " + std::to_string((int)std::lround(p.totalGlobalMem / 1e9)) + " GB, driver " +
                             driver_version() + " | " + cpu_name() + ", " + std::to_string(std::thread::hardware_concurrency()) +
                             " threads | RAM " + std::to_string((int)std::lround(ms.ullTotalPhys / 1e9)) + " GB";
    fprintf(stderr, "\nmoex bench %s\nThis PC: %s | %d expert slots%s\n\n", VERSION, spec.c_str(), eng.nslots, eng.low_ram ? " | low-RAM mode" : "");
    fprintf(stderr, "For a fair number:\n");
#ifdef _WIN32  // shortcut: no battery check on Linux
    SYSTEM_POWER_STATUS ps{};
    if (GetSystemPowerStatus(&ps) && ps.ACLineStatus == 0)
        fprintf(stderr, "  - YOU ARE ON BATTERY: plug the laptop in (GPUs and CPUs run much slower on battery).\n");
#endif
    fprintf(stderr, "  - Close browser tabs, games, Discord and other apps now: they take RAM, VRAM and CPU from moex.\n"
                    "  - Plugged in, Windows power mode \"Best performance\", and the laptop's own performance mode if it has one.\n\n");

    struct Phase { const char* name; bool mtp, fast; };
    std::vector<Phase> phases = {{"warm-up (not counted)", mtp, false}, {"MTP off", false, false}};
    if (mtp) phases.push_back({"MTP on", true, false});
    phases.push_back({mtp ? "MTP on + --fast" : "--fast", mtp, true});
    std::vector<double> avg;
    for (const Phase& ph : phases) {
        fprintf(stderr, "%s:\n", ph.name);
        eng.has_mtp = ph.mtp;
        eng.set_fast(ph.fast);
        double sum = 0;
        for (int i = 0; i < np; i++) {
            eng.reset(); ss.ctx.clear();
            auto prompt = ss.render({{"user", BENCH_PROMPTS[i]}});
            double t0 = now(), t1 = 0; int ntok = 0;
            ss.generate(prompt, BENCH_TOKENS, [&](int) { if (!ntok++) t1 = now(); return true; });
            const double t2 = now(), v = ntok > 1 ? (ntok - 1) / (t2 - t1) : 0.0;
            sum += v;
            fprintf(stderr, "  %d/%d  %5.1f tok/s  %3d tokens  first token %.2f s  %s\n", i + 1, np, v, ntok, t1 - t0, BENCH_PROMPTS[i]);
        }
        if (&ph != &phases[0]) { avg.push_back(sum / np); fprintf(stderr, "  => %s: %.1f tok/s\n", ph.name, sum / np); }
        fprintf(stderr, "\n");
    }
    eng.has_mtp = mtp; eng.set_fast(false);

    std::string res;
    for (size_t i = 0; i < avg.size(); i++) {
        char b[64]; snprintf(b, sizeof b, "%s%s %.1f", i ? ", " : "", phases[i + 1].name, avg[i]);
        res += b;
    }
    fprintf(stderr, "Result (tok/s): %s\n\nShare this line:\n", res.c_str());
    printf("moex %s | %s | %d slots | %s tok/s\n", VERSION, spec.c_str(), eng.nslots, res.c_str());
    fflush(stdout);
    const bool head = std::filesystem::exists(std::filesystem::path(model) / "mtp.safetensors");
    if (!mtp && head && !eng.low_ram)
        fprintf(stderr, "\n(no MTP numbers: --no-mtp)\n");
    else if (!mtp && !eng.low_ram)
        fprintf(stderr, "\n(no MTP numbers: run `moex download` once more to get the MTP head, then `moex bench` again)\n");
    else if (!mtp)
        fprintf(stderr, "\n(no MTP numbers: this PC's free RAM can't hold every expert, where MTP is slower)\n");
    fprintf(stderr, "\nUse the fastest one: `moex chat` (MTP is on by default when it is loaded), `moex chat --no-mtp`, or `moex chat --fast`.\n");
    return 0;
}
