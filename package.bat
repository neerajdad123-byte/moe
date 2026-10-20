@echo off
rem package.bat  -> dist\moex-windows-x64.zip (CUDA 13, driver 580+) and, if CUDA 12.8 is installed,
rem                 dist\moex-windows-x64-cuda12.zip (older drivers). CI builds both on every push.
setlocal
cd /d "%~dp0"
call "%~dp0build.bat" || exit /b 1
if exist dist rmdir /s /q dist
call :zip build\moex.exe moex-windows-x64 || exit /b 1
if defined CUDA_PATH_V12_8 (
  call "%~dp0build.bat" cuda12 || exit /b 1
  call :zip build\cuda12\moex.exe moex-windows-x64-cuda12 || exit /b 1
) else echo CUDA 12.8 not installed: no cuda12 zip (CI makes it)
exit /b 0

:zip
if exist dist\moex rmdir /s /q dist\moex
mkdir dist\moex
copy /y %1 dist\moex\ > nul
copy /y README.md dist\moex\ > nul
copy /y LICENSE dist\moex\ > nul
copy /y data\moex_hot.bin dist\moex\ > nul
copy /y moex-bench.bat dist\moex\ > nul
powershell -NoProfile -Command "Compress-Archive -Path dist\moex\* -DestinationPath dist\%2.zip -Force" || exit /b 1
echo dist\%2.zip
