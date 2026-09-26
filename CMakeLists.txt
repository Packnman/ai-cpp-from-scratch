cmake_minimum_required(VERSION 3.18)

project(ai_cpp VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 11.2 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AI_CPP_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures to build for (for example: 86 or 86;89)")

add_library(ai_cpp STATIC
    lib/src/cli_options.cpp
    lib/src/cuda_function.cpp
    lib/src/cuda_function_Add.cpp
    lib/src/cuda_function_BatchMatMul.cpp
    lib/src/cuda_function_BatchNorm.cpp
    lib/src/cuda_function_Conv2D.cpp
    lib/src/cuda_function_Dropout.cpp
    lib/src/cuda_function_Embedding.cpp
    lib/src/cuda_function_GELU.cpp
    lib/src/cuda_function_IndexCrossEntropy.cu
    lib/src/cuda_function_LayerNorm.cpp
    lib/src/cuda_function_Linear.cpp
    lib/src/cuda_function_Mask.cpp
    lib/src/cuda_function_Permute.cpp
    lib/src/cuda_function_Pooling.cpp
    lib/src/cuda_function_ReLU.cpp
    lib/src/cuda_function_Reshape.cpp
    lib/src/cuda_function_Scale.cpp
    lib/src/cuda_function_Softmax.cpp
    lib/src/cuda_function_SoftmaxCrossEntropy.cpp
    lib/src/cuda_matrix.cpp
    lib/src/cuda_matrix.cu
    lib/src/cuda_memory.cpp
    lib/src/cuda_tensor.cpp
    lib/src/matrix.cpp
    lib/src/matrix_euler.cpp
    lib/src/matrix_quaternion.cpp
    lib/src/module.cpp
    lib/src/optimizer.cpp
    lib/src/optimizer_SGD.cpp
    lib/src/optimizer_adam.cpp
    lib/src/tokenizer_character.cpp
)

target_include_directories(ai_cpp
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(ai_cpp
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
)
target_compile_options(ai_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
set_target_properties(ai_cpp PROPERTIES
    CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}"
)
