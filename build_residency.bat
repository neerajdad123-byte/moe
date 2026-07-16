@echo off
setlocal
cd /d "C:\Users\neera\OneDrive\Desktop\moe"
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
REM nvcc drives cl as host compiler. SM 8.9 = RTX 4050 Ada.
nvcc -std=c++20 -O3 -arch=sm_89 ^
  -I src ^
  -Xcompiler "/EHsc /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267" ^
  src\tools\moex_residency.cu ^
  src\cuda\device_model.cu ^
  src\gguf\gguf.cpp ^
  src\support\mmap_file.cpp ^
  src\model\manifest.cpp ^
  -o build\moex_residency.exe > build\compile_res.log 2>&1
echo NVCC_EXIT=%errorlevel% > build\result_res.log
endlocal
