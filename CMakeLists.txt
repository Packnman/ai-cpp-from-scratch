cmake_minimum_required(VERSION 3.18)

project(ai_cpp LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(AI_CPP_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures to build for (for example: 86 or 86;89)")

add_library(ai_cpp STATIC
    src/checkpoint.cpp
    src/model.cpp
    src/persistence.cpp
    src/cuda_matrix.cpp
    src/cuda_matrix.cu
    src/function.cpp
    src/graph.cpp
    src/matrix.cpp
    src/optimizer.cpp
    src/tensor.cpp
)
target_include_directories(ai_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ai_cpp PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(ai_cpp PROPERTIES
    CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}"
)

add_executable(model_checkpoint_check tests/model_checkpoint_check.cpp)
target_link_libraries(model_checkpoint_check PRIVATE ai_cpp)
set_target_properties(model_checkpoint_check PROPERTIES CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}")

add_executable(cuda_check tests/cuda_check.cu)
target_link_libraries(cuda_check PRIVATE CUDA::cudart)
set_target_properties(cuda_check PROPERTIES
    CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}"
)

if(BUILD_TESTING)
    add_test(NAME cuda_check COMMAND cuda_check)
    add_test(NAME model_checkpoint_check COMMAND model_checkpoint_check)
endif()
