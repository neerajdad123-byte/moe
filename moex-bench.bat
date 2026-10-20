@echo off
rem Double-click: downloads the model once (~20 GB, resumes, skips what is already there), then measures moex on
rem this PC (MTP off / MTP on / --fast) and prints a line to share.
rem Model already somewhere else? From a terminal in this folder:  moex-bench.bat --model D:\path\to\model-folder
cd /d "%~dp0"
echo Close browser tabs, games and other apps first, and plug the laptop in.
echo.
if "%~1"=="" (
  moex.exe download
  if errorlevel 1 goto end
)
moex.exe bench %*
:end
echo.
pause
