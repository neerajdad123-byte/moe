@echo off
rem build.bat   -> build\test_kernels.exe, build\moex.exe
setlocal
cd /d "%~dp0"
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
set NV=nvcc -std=c++20 -O3 -arch=sm_89 -lineinfo -Xcompiler "/EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS"
%NV% tests\test_kernels.cu -o build\test_kernels.exe || exit /b 1
%NV% src\moex.cu -o build\moex.exe || exit /b 1
