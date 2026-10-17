#!/bin/sh
# build.sh       -> build/moex (Linux; RTX 20/30/40/50, A100, H100 + PTX). Needs the CUDA toolkit (nvcc) and g++.
# build.sh dev   -> for this machine's GPU only (fast rebuilds)
set -e
cd "$(dirname "$0")"
mkdir -p build
FAT="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
 -gencode arch=compute_120,code=compute_120"
[ "$1" = "dev" ] && FAT="-arch=native"
nvcc -std=c++20 -O3 -lineinfo -Xcompiler "-mavx2 -mfma -mf16c -pthread" $FAT src/moex.cu -o build/moex
echo build/moex
