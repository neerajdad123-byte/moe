@echo off
rem build.bat         -> build\moex.exe (RTX 20/30/40/50, A100, H100 + PTX), build\test_kernels.exe, build\test_tokenizer.exe, build\ram_balloon.exe
rem build.bat dev     -> moex.exe for this machine's GPU only (fast rebuilds)
rem build.bat cuda12  -> build\cuda12\moex.exe with CUDA 12.8 (runs on drivers older than 580; needs CUDA_PATH_V12_8)
setlocal
cd /d "%~dp0"
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
set NV=nvcc -std=c++20 -O3 -lineinfo -Xcompiler "/EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS /arch:AVX2"
rem the manifest makes UTF-8 the process code page (paths and user names outside the system code page)
set MANI=-Xlinker /MANIFEST:EMBED -Xlinker /MANIFESTINPUT:src\moex.manifest
set FAT=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 ^
 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 ^
 -gencode arch=compute_120,code=compute_120
if "%1"=="cuda12" (
  if not exist build\cuda12 mkdir build\cuda12
  rem CUDA 12.8 predates this Visual Studio: its version check is too strict, the code builds fine
  "%CUDA_PATH_V12_8%\bin\nvcc" -std=c++20 -O3 -lineinfo -allow-unsupported-compiler -Xcompiler "/EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS /arch:AVX2" %FAT% %MANI% src\moex.cu -o build\cuda12\moex.exe || exit /b 1
  exit /b 0
)
if "%1"=="dev" set FAT=-arch=native
%NV% -arch=native tests\test_kernels.cu -o build\test_kernels.exe || exit /b 1
%NV% tests\test_userenv.cu -o build\test_userenv.exe -ladvapi32 -luser32 || exit /b 1
%NV% tests\test_download.cu -o build\test_download.exe -ladvapi32 -luser32 || exit /b 1
%NV% %FAT% %MANI% src\moex.cu -o build\moex.exe || exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /utf-8 /I "%CUDA_PATH%\include" tests\test_tokenizer.cpp /Fe:build\test_tokenizer.exe /Fo:build\ > build\cl_tok.log || exit /b 1
cl /nologo /std:c++20 /O2 /EHsc tests\test_spec.cpp /Fe:build\test_spec.exe /Fo:build\ > build\cl_spec.log || exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /utf-8 tests\test_dense.cpp /Fe:build\test_dense.exe /Fo:build\ > build\cl_dense.log || exit /b 1
cl /nologo /O2 /EHsc tests\ram_balloon.cpp /Fe:build\ram_balloon.exe /Fo:build\ > build\cl_balloon.log || exit /b 1
