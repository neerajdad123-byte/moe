@echo off
setlocal
set "ROOT=C:\Users\neera\OneDrive\Desktop\moe"
cd /d "%ROOT%"
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
cl /nologo /std:c++20 /EHsc /O2 /W4 /permissive- /utf-8 /D_CRT_SECURE_NO_WARNINGS /I src src\tools\moex_g0.cpp src\model\manifest.cpp src\gguf\gguf.cpp src\support\mmap_file.cpp /Fe:build\moex_g0.exe /Fo:build\ > build\compile_g0.log 2>&1
echo CL_EXIT=%errorlevel% > build\result_g0.log
if exist build\moex_g0.exe (echo EXE=YES >> build\result_g0.log) else (echo EXE=NO >> build\result_g0.log)
endlocal
