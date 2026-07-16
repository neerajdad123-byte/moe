@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /nologo /std:c++20 /EHsc /O2 /W4 /permissive- /utf-8 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS /I src src\tools\gguf_dump.cpp src\gguf\gguf.cpp src\support\mmap_file.cpp /Fe:build\gguf_dump.exe /Fo:build\ 2>&1
echo CL_EXIT=%errorlevel%
