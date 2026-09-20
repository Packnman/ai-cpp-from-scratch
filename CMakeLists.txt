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
add_library(brain_common STATIC src/brain/common/Common.cpp)
target_include_directories(brain_common PUBLIC include)
target_compile_options(brain_common PRIVATE -Wall -Wextra -Wpedantic)
add_library(brain_core STATIC
    src/brain/BrainSystem.cpp
    src/brain/logging/LogManager.cpp src/brain/input/InputAdapter.cpp
    src/brain/world/WorldStateManager.cpp src/brain/goal/GoalManager.cpp
    src/brain/constraint/ConstraintManager.cpp
    src/brain/memory/MemoryManager.cpp src/brain/memory/SQLiteMemoryBackend.cpp
    src/brain/policy/PolicyManager.cpp src/brain/planning/Planner.cpp
    src/brain/execution/ExecutionManager.cpp src/brain/postprocess/PostProcessor.cpp
    src/brain/external/ExternalAIAdapter.cpp src/brain/preprocess/Preprocessor.cpp)
target_include_directories(brain_core PUBLIC include third_party/sqlite)
target_link_libraries(brain_core PUBLIC brain_common Threads::Threads)
target_compile_options(brain_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(ner_core STATIC src/ner/entity.cpp src/ner/rule_extractor.cpp src/ner/model_extractor.cpp)
target_include_directories(ner_core PUBLIC include third_party)
target_compile_options(ner_core PRIVATE -Wall -Wextra -Wpedantic)
add_executable(ner_cli train/ner_main.cpp)
target_link_libraries(ner_cli PRIVATE ner_core)

add_library(sqlite3_vendor STATIC third_party/sqlite/sqlite3.c)
target_include_directories(sqlite3_vendor PUBLIC third_party/sqlite)
target_compile_definitions(sqlite3_vendor PRIVATE SQLITE_ENABLE_FTS5 SQLITE_DQS=0 SQLITE_THREADSAFE=1)
target_link_libraries(sqlite3_vendor PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
target_link_libraries(brain_core PUBLIC sqlite3_vendor)
add_library(agent_core STATIC src/agent/agent.cpp src/agent/components.cpp src/agent/context_builder.cpp
    src/agent/discussion.cpp
    src/agent/memory.cpp src/agent/reasoner.cpp src/agent/tools.cpp)
target_include_directories(agent_core PUBLIC include third_party)
target_link_libraries(agent_core PUBLIC sqlite3_vendor ner_core)
target_compile_options(agent_core PRIVATE -Wall -Wextra -Wpedantic)
add_executable(agent_cli app/main.cpp)
target_link_libraries(agent_cli PRIVATE agent_core)
add_executable(discussion_cli train/discussion_main.cpp)
target_link_libraries(discussion_cli PRIVATE agent_core)

if(AI_CPP_ENABLE_SANITIZERS)
    foreach(target brain_common brain_core agent_core agent_cli)
        target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PUBLIC -fsanitize=address,undefined)
    endforeach()
endif()
if(BUILD_TESTING)
    add_executable(brain_common_check tests/brain/brain_common_check.cpp)
    target_link_libraries(brain_common_check PRIVATE brain_common)
    add_test(NAME brain_common_check COMMAND brain_common_check)
    if(AI_CPP_ENABLE_SANITIZERS)
        target_compile_options(brain_common_check PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(brain_common_check PRIVATE -fsanitize=address,undefined)
    endif()
    add_executable(brain_phase2_check tests/brain/brain_phase2_check.cpp)
    target_link_libraries(brain_phase2_check PRIVATE brain_core)
    add_test(NAME brain_phase2_check COMMAND brain_phase2_check)
    add_executable(brain_phase3_4_check tests/brain/brain_phase3_4_check.cpp)
    target_link_libraries(brain_phase3_4_check PRIVATE brain_core)
    add_test(NAME brain_phase3_4_check COMMAND brain_phase3_4_check)
    add_executable(brain_phase5_8_check tests/brain/brain_phase5_8_check.cpp)
    target_link_libraries(brain_phase5_8_check PRIVATE brain_core)
    add_test(NAME brain_phase5_8_check COMMAND brain_phase5_8_check)
    function(add_brain_spec_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE brain_core)
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "brain;spec")
    endfunction()
    add_brain_spec_test(ut_brn_mod_001_input_adapter tests/brain/ut_brn_mod_001_input_adapter.cpp)
    add_brain_spec_test(ut_brn_mod_002_preprocess tests/brain/ut_brn_mod_002_preprocess.cpp)
    add_brain_spec_test(ut_brn_mod_003_world_state_manager tests/brain/ut_brn_mod_003_world_state_manager.cpp)
    add_brain_spec_test(ut_brn_mod_004_goal_manager tests/brain/ut_brn_mod_004_goal_manager.cpp)
    add_brain_spec_test(ut_brn_mod_005_constraint_manager tests/brain/ut_brn_mod_005_constraint_manager.cpp)
    add_brain_spec_test(ut_brn_mod_006_memory_manager tests/brain/ut_brn_mod_006_memory_manager.cpp)
    add_brain_spec_test(ut_brn_mod_007_policy_manager tests/brain/ut_brn_mod_007_policy_manager.cpp)
    add_brain_spec_test(ut_brn_mod_008_planner tests/brain/ut_brn_mod_008_planner.cpp)
    add_brain_spec_test(ut_brn_mod_009_execution_manager tests/brain/ut_brn_mod_009_execution_manager.cpp)
    add_brain_spec_test(ut_brn_mod_010_post_process tests/brain/ut_brn_mod_010_post_process.cpp)
    add_brain_spec_test(ut_brn_mod_011_external_ai_adapter tests/brain/ut_brn_mod_011_external_ai_adapter.cpp)
    add_brain_spec_test(ut_brn_mod_012_log_trace_manager tests/brain/ut_brn_mod_012_log_trace_manager.cpp)
    add_brain_spec_test(it_brain_integration tests/brain/it_brain_integration.cpp)
    add_executable(discussion_check tests/agent/discussion_check.cpp)
    target_link_libraries(discussion_check PRIVATE agent_core)
    add_test(NAME discussion_check COMMAND discussion_check)
    add_executable(discussion_agent_check tests/agent/discussion_agent_check.cpp)
    target_link_libraries(discussion_agent_check PRIVATE agent_core)
    add_test(NAME discussion_agent_check COMMAND discussion_agent_check)
    add_executable(ner_check tests/ner/ner_check.cpp)
    target_link_libraries(ner_check PRIVATE ner_core)
    add_test(NAME ner_check COMMAND ner_check)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME ner_conversion_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/ner/test_convert_ner.py)
        add_test(NAME discussion_data_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/agent/test_generate_discussion_data.py)
    endif()
    add_executable(agent_check tests/agent/agent_check.cpp)
    target_link_libraries(agent_check PRIVATE agent_core)
    add_test(NAME agent_check COMMAND agent_check)
    add_executable(control_flow_check tests/agent/control_flow_check.cpp)
    target_link_libraries(control_flow_check PRIVATE agent_core)
    add_test(NAME control_flow_check COMMAND control_flow_check)
    add_executable(memory_migration_check tests/agent/memory_migration_check.cpp)
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
        src/model/agent_tokenizer.cpp
        src/model/agent_dataset.cpp
        src/model/agent_transformer.cpp
        src/model/model_language_model.cpp
        src/model/jawiki_sharding.cpp
        src/model/agent_training.cpp)
    target_include_directories(agent_model PUBLIC include PRIVATE third_party)
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
        add_executable(agent_model_check tests/model/agent_model_check.cpp)
        target_link_libraries(agent_model_check PRIVATE agent_model)
        add_test(NAME agent_model_check COMMAND agent_model_check)
    endif()
endif()
