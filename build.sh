#!/bin/sh
# build.sh       -> build/moex (Linux; RTX 20/30/40/50, A100, H100 + PTX) and the test programs (as build.bat).
#                   Needs the CUDA toolkit (nvcc) and g++.
# build.sh dev   -> moex for this machine's GPU only (fast rebuilds)
# build.sh moex  -> moex only (the release build)
set -e
cd "$(dirname "$0")"
mkdir -p build
FAT="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
 -gencode arch=compute_120,code=compute_120"
[ "$1" = "dev" ] && FAT="-arch=native"
NV="nvcc -std=c++20 -O3 -lineinfo -Xcompiler -mavx2,-mfma,-mf16c,-pthread"
CXX="g++ -std=c++20 -O2 -mavx2 -mfma -mf16c -pthread"
if [ "$1" != "moex" ]; then
  $NV -arch=native tests/test_kernels.cu -o build/test_kernels
  $NV tests/test_download.cu -o build/test_download
  CUDA_INC="$(dirname "$(dirname "$(command -v nvcc)")")/include"
  $CXX -I "$CUDA_INC" tests/test_tokenizer.cpp -o build/test_tokenizer
  $CXX tests/test_spec.cpp -o build/test_spec
  $CXX tests/test_dense.cpp -o build/test_dense
  $CXX tests/test_template.cpp -o build/test_template
  $CXX tests/test_gguf.cpp -o build/test_gguf
fi
$NV $FAT src/moex.cu -o build/moex
echo build/moex
