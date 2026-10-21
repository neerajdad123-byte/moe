# Contributing to moex

Thanks for helping. Bug reports, benchmark results from your machine, and pull requests are all welcome.

## Report a bug

Open an [issue](https://github.com/neerajdad123-byte/moe/issues/new) with:

- what you ran (the exact command) and what happened, with the error text;
- `moex --version`, your GPU and VRAM, NVIDIA driver version (`nvidia-smi`), CPU, RAM, and Windows version;
- whether it happens every time.

Security problems go through [SECURITY.md](SECURITY.md), not public issues.

## Share a benchmark

Run `moex bench` and paste the line it prints into a
[benchmark report](https://github.com/neerajdad123-byte/moe/issues/new?template=benchmark.yml). Results from GPUs and
CPUs we haven't tested are the most useful.

## Suggest a feature

Open an issue first and describe what you want to do and why. For anything larger than a small fix, agree on the
approach in the issue before writing code, so no work is wasted.

## Pull requests

1. Build from source (see [Building from source](README.md#building-from-source)): Visual Studio 2026 (C++) and
   CUDA 13.x on Windows, `build.bat dev` for a fast build for your GPU.
2. Keep each pull request to one change, and each commit to one change with a message that says what it fixes or adds.
3. Run the checks that cover what you changed. They are listed in the README:
   - kernels: `build\test_kernels.exe`
   - anything that affects the model's output: `build\moex.exe --check ref/eval1` (must say PASS)
   - the server or API: `moex serve`, then `python tools/test_server.py`, `test_tools.py`, `test_security.py`,
     `test_robust.py`
   - speed changes: `moex bench` before and after, on the same machine, plugged in, nothing else running
4. Say in the pull request what you changed, why, and which checks you ran with their results. For speed changes,
   include the before / after numbers.
5. Match the style of the code around your change.

CI builds both release zips and runs the checks that need no GPU on every pull request. The GPU checks run on the
maintainer's machine before a release.

## License

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE) of this project.
