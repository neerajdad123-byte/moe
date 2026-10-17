# moex installer for Windows. In PowerShell:
#   irm https://raw.githubusercontent.com/neerajdad123-byte/moe/main/install.ps1 | iex
# Installs the latest release to %LOCALAPPDATA%\Programs\moex (no admin) and adds it to your PATH, so `moex` works in
# any terminal. Run it again to update. Uninstall: delete that folder and remove it from PATH (Settings > "environment").
param([string]$Dir = "$env:LOCALAPPDATA\Programs\moex", [switch]$NoPath)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest is many times slower with the progress bar

if (Get-Process moex -ErrorAction SilentlyContinue) { throw "moex is running: close it first, then run the installer again." }
# The main build uses CUDA 13 (driver 580+); older drivers get the CUDA 12 build
$name = 'moex-windows-x64'
$drv = try { [double](& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1) } catch { 0 }
if ($drv -gt 0 -and $drv -lt 580) { $name = 'moex-windows-x64-cuda12'; Write-Host "NVIDIA driver $drv (older than 580): using the CUDA 12 build" }
$zip = Join-Path $env:TEMP "$name.zip"
Write-Host "Downloading moex..."
Invoke-WebRequest "https://github.com/neerajdad123-byte/moe/releases/latest/download/$name.zip" -OutFile $zip -UseBasicParsing
New-Item -ItemType Directory -Force $Dir | Out-Null
Expand-Archive $zip $Dir -Force
Remove-Item $zip

if (-not $NoPath) {
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @(if ($user) { $user -split ';' | Where-Object { $_ } })  # @(): one entry must stay a list, not a string
    if ($parts -notcontains $Dir) { [Environment]::SetEnvironmentVariable('Path', (($parts + $Dir) -join ';'), 'User') }
    if (($env:Path -split ';') -notcontains $Dir) { $env:Path += ";$Dir" }  # this window too
}

Write-Host "Installed to $Dir"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "No NVIDIA driver found. moex needs an NVIDIA GPU (RTX 20 series or newer) and its driver:" -ForegroundColor Yellow
    Write-Host "  https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Next, in any terminal:"
Write-Host "  moex download      (the model, ~22 GB, once)"
Write-Host "  moex chat --load"
