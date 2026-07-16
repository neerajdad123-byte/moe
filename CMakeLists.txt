# MoEx — clean-room MoE expert-paging engine for Qwen3-30B-A3B Q4_K_M
# Implementation rule (design doc §1): MoEx-owned C++/CUDA only.
# No llama.cpp/ggml source, headers, kernels, links, callbacks, or runtime.
cmake_minimum_required(VERSION 3.24)

project(moex LANGUAGES CXX)

option(MOEX_ENABLE_CUDA "Build CUDA resident-decode path (G2+)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive- /Zc:preprocessor /utf-8 /EHsc)
  # Large file offsets: MSVC uses 64-bit std::streamoff by default; keep it explicit.
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# ---- core library (host, no CUDA) --------------------------------------------
add_library(moex_core
  src/support/bytes.cpp
  src/gguf/gguf_reader.cpp
  src/model/manifest.cpp
  src/model/ledger.cpp
)
target_include_directories(moex_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- inspection / feasibility tool (G0) --------------------------------------
add_executable(moex_inspect tools/moex_inspect.cpp)
target_link_libraries(moex_inspect PRIVATE moex_core)

# ---- tests -------------------------------------------------------------------
enable_testing()
add_executable(moex_tests
  tests/test_main.cpp
  tests/test_gguf_reader.cpp
  tests/test_ledger.cpp
)
target_link_libraries(moex_tests PRIVATE moex_core)
add_test(NAME moex_tests COMMAND moex_tests)

if(MOEX_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4050 Laptop = Ada = SM 8.9
  message(STATUS "MoEx: CUDA path enabled (SM 8.9)")
endif()
