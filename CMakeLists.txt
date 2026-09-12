cmake_minimum_required(VERSION 3.18)

project(ai_cpp LANGUAGES CXX CUDA)

find_package(CUDAToolkit 11.2 REQUIRED)
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
    lib/src/tokenizer_character.cpp
    lib/src/cuda_memory.cpp
    lib/src/cuda_matrix.cpp
    lib/src/cuda_matrix.cu
    lib/src/cuda_function.cpp
    lib/src/cuda_function_Add.cpp
    lib/src/cuda_function_BatchMatMul.cpp
    lib/src/cuda_function_BatchNorm.cpp
    lib/src/cuda_function_Conv2D.cpp
    lib/src/cuda_function_Dropout.cpp
    lib/src/cuda_function_Embedding.cpp
    lib/src/cuda_function_GELU.cpp
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
    lib/src/cuda_tensor.cpp
    lib/src/module.cpp
    lib/src/matrix.cpp
    lib/src/matrix_euler.cpp
    lib/src/matrix_quaternion.cpp
    lib/src/optimizer.cpp
    lib/src/optimizer_SGD.cpp
    lib/src/optimizer_adam.cpp
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

if(BUILD_TESTING)
    add_executable(cuda_memory_check tests/cuda_memory_check.cpp)
    target_link_libraries(cuda_memory_check PRIVATE ai_cpp)
    add_test(NAME cuda_memory_check COMMAND cuda_memory_check)
    add_executable(character_tokenizer_check tests/character_tokenizer_check.cpp)
    target_include_directories(character_tokenizer_check PRIVATE ${PROJECT_SOURCE_DIR}/lib/include)
    target_link_libraries(character_tokenizer_check PRIVATE ai_cpp)

    add_executable(embedding_check tests/embedding_check.cpp)
    target_include_directories(embedding_check PRIVATE ${PROJECT_SOURCE_DIR}/lib/include)
    target_link_libraries(embedding_check PRIVATE ai_cpp)

    add_executable(transformer_functions_check tests/transformer_functions_check.cpp)
    target_include_directories(transformer_functions_check PRIVATE ${PROJECT_SOURCE_DIR}/lib/include)
    target_link_libraries(transformer_functions_check PRIVATE ai_cpp)

    add_executable(cuda_index_tensor_check tests/cuda_index_tensor_check.cpp)
    target_include_directories(cuda_index_tensor_check PRIVATE ${PROJECT_SOURCE_DIR}/lib/include)
    target_link_libraries(cuda_index_tensor_check PRIVATE ai_cpp)

    add_executable(row_major_check tests/row_major_check.cpp)
    target_include_directories(row_major_check PRIVATE ${PROJECT_SOURCE_DIR}/lib/include)
    target_link_libraries(row_major_check PRIVATE ai_cpp)

    add_executable(multi_output_check tests/multi_output_check.cpp)
    target_include_directories(multi_output_check PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(multi_output_check PRIVATE ai_cpp)

    add_executable(model_state_check
        tests/model_state_check.cpp
        tests/fixtures/src/neuralnet_mnist.cpp
    )
    target_include_directories(model_state_check PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/fixtures/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(model_state_check PRIVATE ai_cpp)

    add_executable(dropout_check
        tests/dropout_check.cpp
        tests/fixtures/src/neuralnet_mnist.cpp
    )
    target_include_directories(dropout_check PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/fixtures/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(dropout_check PRIVATE ai_cpp)

    add_executable(batchnorm_check tests/batchnorm_check.cpp)
    target_include_directories(batchnorm_check PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(batchnorm_check PRIVATE ai_cpp)

    add_executable(conv2d_check tests/conv2d_check.cpp tests/fixtures/src/neuralnet_cifar10.cpp)
    target_include_directories(conv2d_check PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/fixtures/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(conv2d_check PRIVATE ai_cpp)

    add_executable(maxpool2d_check tests/maxpool2d_check.cpp)
    target_include_directories(maxpool2d_check PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(maxpool2d_check PRIVATE ai_cpp)

    add_executable(cifar10_dataset_check
        tests/cifar10_dataset_check.cpp
        tests/fixtures/src/dataset_cifar10.cpp
    )
    target_include_directories(cifar10_dataset_check PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/fixtures/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(cifar10_dataset_check PRIVATE ai_cpp)

    add_executable(cifar10_model_check
        tests/cifar10_model_check.cpp
        tests/fixtures/src/neuralnet_cifar10.cpp
    )
    target_include_directories(cifar10_model_check PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/fixtures/include
        ${PROJECT_SOURCE_DIR}/lib/include
    )
    target_link_libraries(cifar10_model_check PRIVATE ai_cpp)

    add_test(NAME cuda_check COMMAND cuda_check)
    add_test(NAME character_tokenizer_check COMMAND character_tokenizer_check)
    add_test(NAME embedding_check COMMAND embedding_check)
    add_test(NAME transformer_functions_check COMMAND transformer_functions_check)
    add_test(NAME cuda_index_tensor_check COMMAND cuda_index_tensor_check)
    add_test(NAME row_major_check COMMAND row_major_check)
    add_test(NAME multi_output_check COMMAND multi_output_check)
    add_test(NAME model_state_check COMMAND model_state_check)
    add_test(NAME dropout_check COMMAND dropout_check)
    add_test(NAME batchnorm_check COMMAND batchnorm_check)
    add_test(NAME conv2d_check COMMAND conv2d_check)
    add_test(NAME maxpool2d_check COMMAND maxpool2d_check)
    add_test(NAME cifar10_dataset_check COMMAND cifar10_dataset_check)
    add_test(NAME cifar10_model_check COMMAND cifar10_model_check)
endif()

# Shared conversation data/model and two application entry points.
# Vendored nlohmann/json 3.12.0 (MIT); no download is required at build time.
target_sources(ai_cpp PRIVATE lib/src/cuda_function_IndexCrossEntropy.cu)
add_library(conversation_data STATIC
    model/src/tokenizer_conversation.cpp
    model/src/dataset_conversation.cpp
    lib/src/tokenizer_character.cpp
)
target_include_directories(conversation_data PUBLIC model/include lib/include PRIVATE third_party)
add_library(conversation_runtime STATIC
    model/src/module_Attention.cpp
    model/src/module_FeedForward.cpp
    model/src/model_transformer.cpp
    model/src/conversation_runtime.cpp
)
target_include_directories(conversation_runtime PUBLIC model/include PRIVATE third_party)
target_link_libraries(conversation_runtime PUBLIC ai_cpp conversation_data)

add_library(conversation_training STATIC train/src/train.cpp)
target_include_directories(conversation_training PUBLIC train/include PRIVATE third_party)
target_link_libraries(conversation_training PUBLIC conversation_runtime)
add_executable(main_train train/main_train.cpp)
target_link_libraries(main_train PRIVATE conversation_training)

add_library(conversation_validation STATIC
    validation/src/validation.cpp)
target_include_directories(conversation_validation PUBLIC validation/include PRIVATE third_party)
target_link_libraries(conversation_validation PUBLIC conversation_runtime)
add_executable(main_validation validation/main_validation.cpp)
target_link_libraries(main_validation PRIVATE conversation_validation)

if(BUILD_TESTING)
    add_executable(conversation_check tests/conversation_check.cpp)
    target_include_directories(conversation_check PRIVATE third_party)
    target_link_libraries(conversation_check PRIVATE conversation_training conversation_validation)
    add_test(NAME conversation_check COMMAND conversation_check)
    add_executable(cuda_memory_numerics tests/cuda_memory_numerics.cpp)
    target_link_libraries(cuda_memory_numerics PRIVATE conversation_runtime)
    add_test(NAME cuda_memory_compare COMMAND ${CMAKE_COMMAND}
        -DPROGRAM=$<TARGET_FILE:cuda_memory_numerics>
        -P ${PROJECT_SOURCE_DIR}/tests/cuda_memory_compare.cmake)
    add_executable(cuda_memory_benchmark tests/cuda_memory_benchmark.cpp)
    target_link_libraries(cuda_memory_benchmark PRIVATE conversation_training)
endif()
