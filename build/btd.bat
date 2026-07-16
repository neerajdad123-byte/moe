@echo off
setlocal
cd /d "C:\Users\neera\OneDrive\Desktop\moe"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
del /q build\*.obj 2>nul
cl /nologo /std:c++20 /EHsc /O2 /permissive- /utf-8 /D_CRT_SECURE_NO_WARNINGS /I src ^
  src\tools\test_detok.cpp src\model\tokenizer.cpp src\gguf\gguf.cpp src\support\mmap_file.cpp ^
  /Fe:build\test_detok.exe /Fo:build\ > build\compile_td2.log 2>&1
echo CL_EXIT=%errorlevel% > build\result_td2.log
endlocal
