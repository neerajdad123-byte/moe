@echo off
rem package.bat  -> dist\moex-windows-x64.zip  (release build of moex.exe + README)
setlocal
cd /d "%~dp0"
call "%~dp0build.bat" || exit /b 1
if exist dist rmdir /s /q dist
mkdir dist\moex
copy /y build\moex.exe dist\moex\ > nul
copy /y README.md dist\moex\ > nul
copy /y LICENSE dist\moex\ > nul
copy /y data\moex_hot.bin dist\moex\ > nul
powershell -NoProfile -Command "Compress-Archive -Path dist\moex\* -DestinationPath dist\moex-windows-x64.zip -Force"
echo dist\moex-windows-x64.zip
