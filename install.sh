#!/bin/sh
# moex installer for Linux (x86-64):
#   curl -fsSL https://raw.githubusercontent.com/neerajdad123-byte/moe/main/install.sh | sh
# Installs the latest release to ~/.local/share/moex (no sudo) and links ~/.local/bin/moex to it, so `moex` works in
# any terminal. The model downloads next to the real binary, not into ~/.local/bin. Run it again to update.
# Uninstall: rm -rf ~/.local/share/moex ~/.local/bin/moex
set -e
DIR="${MOEX_DIR:-$HOME/.local/share/moex}"
# CUDA 13 build for driver 580+, else the CUDA 12 one (driver 570+)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
TAR=moex-linux-x64; [ -n "$DRV" ] && [ "$DRV" -lt 580 ] 2>/dev/null && TAR=moex-linux-x64-cuda12
URL="${MOEX_URL:-https://github.com/neerajdad123-byte/moe/releases/latest/download/$TAR.tar.gz}"

[ "$(uname -m)" = x86_64 ] || { echo "moex needs an x86-64 Linux PC (this is $(uname -m))." >&2; exit 1; }
if pgrep -x moex >/dev/null 2>&1; then echo "moex is running: close it first, then run the installer again." >&2; exit 1; fi

echo "Downloading moex..."
TMPF=$(mktemp); trap 'rm -f "$TMPF"' EXIT
# to a file first: in a pipe, sh would hide curl's error behind tar's
curl -fsSL -o "$TMPF" "$URL" || { echo "Cannot download $URL. If the latest release has no Linux build yet, build moex from source:
  https://github.com/neerajdad123-byte/moe#building-from-source" >&2; exit 1; }
mkdir -p "$DIR"
tar -xzf "$TMPF" -C "$DIR" --strip-components=1  # the tarball holds moex/{moex,moex_hot.bin,README.md,LICENSE}
"$DIR/moex" install  # the ~/.local/bin/moex link

echo "Installed to $DIR"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA driver found. moex needs an NVIDIA GPU (RTX 20 series or newer) and its driver."
fi
echo
echo "Next, in any terminal:"
echo "  moex download      (the model, ~22 GB, once)"
echo "  moex chat --load"
