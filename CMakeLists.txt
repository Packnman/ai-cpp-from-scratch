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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AI_CPP_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures to build for (for example: 86 or 86;89)")

add_library(ai_cpp STATIC
    lib/src/cuda_matrix.cpp
    lib/src/cuda_matrix.cu
    lib/src/cuda_function.cpp
    lib/src/cuda_tensor.cpp
    lib/src/module.cpp
    lib/src/matrix.cpp
    lib/src/optimizer.cpp
)
target_include_directories(ai_cpp PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_link_libraries(ai_cpp PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
set_target_properties(ai_cpp PROPERTIES
    CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}"
)

add_executable(cuda_check tests/cuda_check.cu)
target_link_libraries(cuda_check PRIVATE CUDA::cudart)
set_target_properties(cuda_check PROPERTIES
    CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}"
)

add_executable(mnist_train
    src/main.cpp
    src/mnist.cpp
    src/neuralnet.cpp
)
target_include_directories(mnist_train PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/lib/include
)
target_link_libraries(mnist_train PRIVATE ai_cpp)

if(BUILD_TESTING)
    add_executable(multi_output_check tests/multi_output_check.cpp)
    target_include_directories(multi_output_check PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(multi_output_check PRIVATE ai_cpp)

    add_executable(model_state_check
        tests/model_state_check.cpp
        src/neuralnet.cpp
    )
    target_include_directories(model_state_check PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(model_state_check PRIVATE ai_cpp)

    add_executable(dropout_check
        tests/dropout_check.cpp
        src/neuralnet.cpp
    )
    target_include_directories(dropout_check PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(dropout_check PRIVATE ai_cpp)

    add_test(NAME cuda_check COMMAND cuda_check)
    add_test(NAME multi_output_check COMMAND multi_output_check)
    add_test(NAME model_state_check COMMAND model_state_check)
    add_test(NAME dropout_check COMMAND dropout_check)
endif()
