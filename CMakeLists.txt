cmake_minimum_required(VERSION 3.18)

project(ai_cpp LANGUAGES CXX CUDA)

# C++ / CUDA standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3060 Ti = Compute Capability 8.6
set(CMAKE_CUDA_ARCHITECTURES 86)

# 実行ファイル
add_executable(ai_cpp
    tests/main.cpp
)

# include/
target_include_directories(ai_cpp
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# CUDAライブラリ
find_package(CUDAToolkit REQUIRED)

target_link_libraries(ai_cpp
    PRIVATE
    CUDA::cudart
    CUDA::cublas
)