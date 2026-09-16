cmake_minimum_required(VERSION 3.18)
project(ai_cpp_agent LANGUAGES C CXX)
include(CTest)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(AI_CPP_BUILD_CUDA_LIB "Build the legacy lib/ CUDA library" OFF)
option(AI_CPP_BUILD_MODEL "Build the CUDA Agent Transformer and training tools" OFF)
option(AI_CPP_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
set(AI_CPP_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

find_package(Threads REQUIRED)
add_library(sqlite3_vendor STATIC third_party/sqlite/sqlite3.c)
target_include_directories(sqlite3_vendor PUBLIC third_party/sqlite)
target_compile_definitions(sqlite3_vendor PRIVATE SQLITE_ENABLE_FTS5 SQLITE_DQS=0 SQLITE_THREADSAFE=1)
target_link_libraries(sqlite3_vendor PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
add_library(agent_core STATIC src/agent/agent.cpp src/agent/components.cpp src/agent/context_builder.cpp
    src/agent/memory.cpp src/agent/reasoner.cpp src/agent/tools.cpp)
target_include_directories(agent_core PUBLIC include third_party)
target_link_libraries(agent_core PUBLIC sqlite3_vendor)
target_compile_options(agent_core PRIVATE -Wall -Wextra -Wpedantic)
add_executable(agent_cli app/main.cpp)
target_link_libraries(agent_cli PRIVATE agent_core)

if(AI_CPP_ENABLE_SANITIZERS)
    foreach(target agent_core agent_cli)
        target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PUBLIC -fsanitize=address,undefined)
    endforeach()
endif()
if(BUILD_TESTING)
    add_executable(agent_check tests/agent_check.cpp)
    target_link_libraries(agent_check PRIVATE agent_core)
    add_test(NAME agent_check COMMAND agent_check)
    add_executable(control_flow_check tests/control_flow_check.cpp)
    target_link_libraries(control_flow_check PRIVATE agent_core)
    add_test(NAME control_flow_check COMMAND control_flow_check)
    add_executable(memory_migration_check tests/memory_migration_check.cpp)
    target_link_libraries(memory_migration_check PRIVATE agent_core)
    add_test(NAME memory_migration_check COMMAND memory_migration_check)
    if(AI_CPP_ENABLE_SANITIZERS)
        target_compile_options(agent_check PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(agent_check PRIVATE -fsanitize=address,undefined)
    endif()
endif()

if(AI_CPP_BUILD_CUDA_LIB OR AI_CPP_BUILD_MODEL)
    enable_language(CUDA)
    find_package(CUDAToolkit 11.2 REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_library(ai_cpp STATIC
        lib/src/cli_options.cpp lib/src/tokenizer_character.cpp lib/src/cuda_memory.cpp
        lib/src/cuda_matrix.cpp lib/src/cuda_matrix.cu lib/src/cuda_function.cpp
        lib/src/cuda_function_Add.cpp lib/src/cuda_function_BatchMatMul.cpp
        lib/src/cuda_function_BatchNorm.cpp lib/src/cuda_function_Conv2D.cpp
        lib/src/cuda_function_Dropout.cpp lib/src/cuda_function_Embedding.cpp
        lib/src/cuda_function_GELU.cpp lib/src/cuda_function_IndexCrossEntropy.cu
        lib/src/cuda_function_LayerNorm.cpp lib/src/cuda_function_Linear.cpp
        lib/src/cuda_function_Mask.cpp lib/src/cuda_function_Permute.cpp
        lib/src/cuda_function_Pooling.cpp lib/src/cuda_function_ReLU.cpp
        lib/src/cuda_function_Reshape.cpp lib/src/cuda_function_Scale.cpp
        lib/src/cuda_function_Softmax.cpp lib/src/cuda_function_SoftmaxCrossEntropy.cpp
        lib/src/cuda_tensor.cpp lib/src/module.cpp lib/src/matrix.cpp lib/src/matrix_euler.cpp
        lib/src/matrix_quaternion.cpp lib/src/optimizer.cpp lib/src/optimizer_SGD.cpp lib/src/optimizer_adam.cpp)
    target_include_directories(ai_cpp PUBLIC lib/include)
    target_link_libraries(ai_cpp PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
    set_target_properties(ai_cpp PROPERTIES CUDA_ARCHITECTURES "${AI_CPP_CUDA_ARCHITECTURES}")
endif()

if(AI_CPP_BUILD_MODEL)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SPM_BUILD_TCMALLOC OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/sentencepiece EXCLUDE_FROM_ALL)
    add_library(agent_model STATIC
        model/src/agent_tokenizer.cpp
        model/src/agent_dataset.cpp
        model/src/agent_transformer.cpp
        model/src/model_language_model.cpp
        model/src/jawiki_sharding.cpp
        model/src/agent_training.cpp)
    target_include_directories(agent_model PUBLIC include model/include PRIVATE third_party)
    target_include_directories(agent_model SYSTEM PRIVATE
        third_party/sentencepiece/src
        third_party/sentencepiece/src/builtin_pb
        third_party/sentencepiece/third_party/protobuf-lite)
    target_link_libraries(agent_model PUBLIC ai_cpp agent_core sentencepiece_train-static)
    target_compile_options(agent_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(agent_cli PRIVATE AI_CPP_BUILD_MODEL=1)
    target_link_libraries(agent_cli PRIVATE agent_model)
    add_executable(agent_model_cli train/agent_model_main.cpp)
    target_link_libraries(agent_model_cli PRIVATE agent_model)
    if(BUILD_TESTING)
        add_executable(agent_model_check tests/agent_model_check.cpp)
        target_link_libraries(agent_model_check PRIVATE agent_model)
        add_test(NAME agent_model_check COMMAND agent_model_check)
    endif()
endif()
