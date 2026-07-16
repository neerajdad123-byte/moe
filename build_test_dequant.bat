@echo off
setlocal
cd /d "C:\Users\neera\OneDrive\Desktop\moe"
if not exist build mkdir build
if not exist build\td mkdir build\td
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > build\vcvars.log 2>&1
cl /nologo /std:c++20 /EHsc /O2 /W4 /permissive- /utf-8 /D_CRT_SECURE_NO_WARNINGS /I src src\tools\test_dequant.cpp src\compute\dequant.cpp /Fe:build\test_dequant.exe /Fo:build\td\ > build\compile_td.log 2>&1
echo CL_EXIT=%errorlevel% > build\result_td.log
endlocal
