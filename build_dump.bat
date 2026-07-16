@echo off
setlocal
set "ROOT=C:\Users\neera\OneDrive\Desktop\moe"
cd /d "%ROOT%"
if not exist build mkdir build
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
cl /nologo /std:c++20 /EHsc /O2 /W4 /permissive- /utf-8 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /I src src\tools\gguf_dump.cpp src\gguf\gguf.cpp src\support\mmap_file.cpp /Fe:build\gguf_dump.exe /Fo:build\ > build\compile.log 2>&1
echo CL_EXIT=%errorlevel% > build\result.log
if exist build\gguf_dump.exe (echo EXE=YES >> build\result.log) else (echo EXE=NO >> build\result.log)
endlocal
