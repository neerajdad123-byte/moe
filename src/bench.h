// `moex bench`: speed on the kind of prompt most people type: a short everyday question answered in a few
// paragraphs. One kind on purpose (no code/math/translation mix), so the number matches what chat feels like.
// Each question is a new conversation with chat's sampling and a fixed seed. None of these are in data/prompts.txt
// (the prompts behind the shipped moex_hot.bin). Included by moex.cu after session.h.
#pragma once

static const char* BENCH_PROMPTS[] = {
    "Why do we get hiccups, and what actually helps stop them?",
    "How does a microwave oven heat food?",
    "Why do onions make people cry when they cut them?",
    "How do noise-cancelling headphones work?",
    "Why does bread go stale, and how can I keep it fresh longer?",
    "How do vaccines teach the body to fight a disease?",
    "Why do cats purr?",
    "How does GPS on a phone know where I am?",
    "Why is it harder to fall asleep after looking at a phone screen at night?",
    "How do airplanes stay in the air?",
};
constexpr int BENCH_TOKENS = 256;  // reply cap: a few paragraphs, like a typical chat answer

static std::string cpu_name() {
    int r[4]; char s[49] = {};
    for (int i = 0; i < 3; i++) { __cpuid(r, 0x80000002 + i); memcpy(s + 16 * i, r, 16); }
    std::string n = s;
    n.erase(0, n.find_first_not_of(' '));
    return n;
}

static int run_bench(Engine& eng, Tokenizer& tk, const std::string& model) {
    Session ss(eng, tk);
    const int np = (int)(sizeof BENCH_PROMPTS / sizeof *BENCH_PROMPTS);
    fprintf(stderr, "\nmoex bench: %d everyday questions, up to %d tokens each, a new conversation for each\n\n", np, BENCH_TOKENS);
    std::vector<double> tps, ttft;
    for (int i = 0; i < np; i++) {
        eng.reset(); ss.ctx.clear();
        auto prompt = ss.render({{"user", BENCH_PROMPTS[i]}});
        double t0 = now(), t1 = 0; int ntok = 0;
        ss.generate(prompt, BENCH_TOKENS, [&](int) { if (!ntok++) t1 = now(); return true; });
        const double t2 = now(), v = ntok > 1 ? (ntok - 1) / (t2 - t1) : 0.0;
        tps.push_back(v); ttft.push_back(t1 - t0);
        fprintf(stderr, "  %2d/%d  %5.1f tok/s  %3d tokens  first token %.2f s  %s\n", i + 1, np, v, ntok, t1 - t0, BENCH_PROMPTS[i]);
    }
    double avg = 0; for (double v : tps) avg += v; avg /= np;
    const double lo = *std::min_element(tps.begin(), tps.end()), hi = *std::max_element(tps.begin(), tps.end());
    const double f_lo = *std::min_element(ttft.begin(), ttft.end()), f_hi = *std::max_element(ttft.begin(), ttft.end());
    fprintf(stderr, "\nResult: average %.1f tok/s | low %.1f | high %.1f   (first token %.2f-%.2f s)\n", avg, lo, hi, f_lo, f_hi);

    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p, 0));
    MEMORYSTATUSEX ms{sizeof ms}; GlobalMemoryStatusEx(&ms);
    fprintf(stderr, "\nShare it: open https://github.com/neerajdad123-byte/moe/issues/new?template=benchmark.yml and paste this line:\n");
    printf("moex %s | %s %.1f GB | %s | RAM %.0f GB | %d expert slots%s%s | avg %.1f, low %.1f, high %.1f tok/s\n", VERSION, p.name,
           p.totalGlobalMem / 1e9, cpu_name().c_str(), ms.ullTotalPhys / 1e9, eng.nslots, eng.has_mtp ? " | MTP on" : "",
           eng.low_ram ? " | low-RAM mode" : "", avg, lo, hi);
    fflush(stdout);

    fprintf(stderr, "\nTips to go faster:\n");
    SYSTEM_POWER_STATUS ps{};
    if (GetSystemPowerStatus(&ps) && ps.ACLineStatus == 0)
        fprintf(stderr, "  - You are on battery: plug the laptop in. GPUs and CPUs run much slower on battery.\n");
    fprintf(stderr, "  - Windows power mode \"Best performance\" (Settings > System > Power), and the laptop's own performance mode\n"
                    "    if it has one (e.g. Armoury Crate, Legion Space, Omen Gaming Hub).\n");
    fprintf(stderr, "  - Close browser tabs and apps you don't need (games, video, Discord) before starting moex: they eat RAM and\n"
                    "    VRAM, and moex turns all free VRAM into expert cache (more cache is faster).\n");
    if (eng.low_ram)
        fprintf(stderr, "  - Your free RAM can't hold every expert, so some are read from the SSD: every extra GB of free RAM is\n"
                        "    about +2 tok/s here.\n");
    if (!eng.has_mtp && !eng.low_ram)
        fprintf(stderr, "  - Try multi-token prediction: `moex bench --mtp` (then `moex chat --mtp`), often ~10%% faster on this PC%s.\n",
                std::filesystem::exists(std::filesystem::path(model) / "mtp.safetensors") ? "" : " (run `moex download` first to get the MTP head)");
    else if (eng.has_mtp)
        fprintf(stderr, "  - Compare without MTP: `moex bench`, and keep --mtp in chat only if it was faster here%s.\n",
                eng.low_ram ? " (in low-RAM mode its extra experts are extra SSD reads, often slower)" : "");
    return 0;
}
