@echo off
rem build.bat         -> build\moex.exe (RTX 20/30/40/50, A100, H100 + PTX), build\test_kernels.exe, build\test_tokenizer.exe
rem build.bat dev     -> moex.exe for this machine's GPU only (fast rebuilds)
setlocal
cd /d "%~dp0"
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
set NV=nvcc -std=c++20 -O3 -lineinfo -Xcompiler "/EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS /arch:AVX2"
set FAT=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 ^
 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 ^
 -gencode arch=compute_120,code=compute_120
if "%1"=="dev" set FAT=-arch=native
%NV% -arch=native tests\test_kernels.cu -o build\test_kernels.exe || exit /b 1
%NV% %FAT% src\moex.cu -o build\moex.exe || exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /utf-8 /I "%CUDA_PATH%\include" tests\test_tokenizer.cpp /Fe:build\test_tokenizer.exe /Fo:build\ > build\cl_tok.log || exit /b 1
