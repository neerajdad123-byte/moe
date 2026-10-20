@echo off
rem Double-click: downloads the model once (~20 GB, resumes if interrupted, skips what is already there),
rem then measures moex on this PC (MTP off / MTP on / --fast) and prints a line to share.
cd /d "%~dp0"
echo Close browser tabs, games and other apps first, and plug the laptop in.
echo.
moex.exe download
if errorlevel 1 goto end
moex.exe bench
:end
echo.
pause
