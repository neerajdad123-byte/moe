@echo off
cd /d "%~dp0.."
if not exist build mkdir build
if not exist build\bench50 mkdir build\bench50
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
nvcc -std=c++20 -O3 -arch=sm_89 -I src ^
  -Xcompiler "/EHsc /D_CRT_SECURE_NO_WARNINGS /wd4244 /wd4267" ^
  src\tools\moex_bench50.cu ^
  src\cuda\forward.cu ^
  src\cuda\device_model.cu ^
  src\gguf\gguf.cpp ^
  src\support\mmap_file.cpp ^
  src\model\manifest.cpp ^
  src\model\tokenizer.cpp ^
  -o build\moex_bench50.exe > build\compile_bench50.log 2>&1
echo NVCC_EXIT=%errorlevel% > build\result_bench50.log
