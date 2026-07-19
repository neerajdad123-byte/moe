@echo off
cd /d "%~dp0.."
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
nvcc -std=c++20 -O3 -arch=sm_89 -I src ^
  -Xcompiler "/EHsc /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267" ^
  src\tools\moex_bwprobe.cu ^
  -o build\moex_bwprobe.exe > build\compile_bwprobe.log 2>&1
echo NVCC_EXIT=%errorlevel% > build\result_bwprobe.log
