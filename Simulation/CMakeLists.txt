add_library(simulation_core STATIC
    src/SimulationManager.cpp
    src/MuJoCoModel.cpp
    src/MuJoCoPlant.cpp
    src/MuJoCoActuatorAdapter.cpp
    src/MuJoCoStateAdapter.cpp
    src/MuJoCoContactManager.cpp
    src/GroundTruthSensorSimulator.cpp
    src/SimulationLog.cpp)
target_include_directories(simulation_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(simulation_core PUBLIC actuator_sim mujoco::mujoco)
target_compile_options(simulation_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(simulation_core PUBLIC
    AI_CPP_MUJOCO_VERSION="${AI_CPP_MUJOCO_VERSION}")

add_executable(simulation_cli src/simulation_main.cpp)
target_link_libraries(simulation_cli PRIVATE simulation_core)
target_compile_definitions(simulation_cli PRIVATE
    AI_CPP_SIMULATION_DEFAULT_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/model/test/single_joint.xml")

if(AI_CPP_ENABLE_MUJOCO_VIEWER)
    find_package(glfw3 ${AI_CPP_GLFW_VERSION} EXACT CONFIG QUIET)
    if(NOT TARGET glfw)
        if(NOT AI_CPP_FETCH_GLFW)
            message(FATAL_ERROR
                "GLFW ${AI_CPP_GLFW_VERSION} was not found; install it or enable AI_CPP_FETCH_GLFW")
        endif()
        include(FetchContent)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG ${AI_CPP_GLFW_VERSION}
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(glfw)
    endif()
    target_sources(simulation_core PRIVATE src/MuJoCoViewer.cpp)
    target_link_libraries(simulation_core PRIVATE glfw)
    target_compile_definitions(simulation_cli PRIVATE AI_CPP_MUJOCO_VIEWER=1)
endif()

if(AI_CPP_ENABLE_SANITIZERS)
    foreach(target simulation_core simulation_cli)
        target_compile_options(${target} PUBLIC
            -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PUBLIC -fsanitize=address,undefined)
    endforeach()
endif()

if(BUILD_TESTING)
    function(add_simulation_test target source labels)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE simulation_core)
        target_compile_definitions(${target} PRIVATE
            AI_CPP_SIMULATION_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/model")
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES LABELS "${labels}")
    endfunction()

    add_simulation_test(ut_simulation_manager
        test/unit/UT-SIM-MOD-001_simulation_manager.cpp "simulation;unit")
    add_simulation_test(ut_mujoco_plant
        test/unit/UT-SIM-MOD-002_mujoco_plant.cpp "simulation;unit;mujoco")
    add_simulation_test(ut_actuator_adapter
        test/unit/UT-SIM-MOD-004_actuator_adapter.cpp "simulation;unit;mujoco")
    add_simulation_test(ut_state_adapter
        test/unit/UT-SIM-MOD-005_state_adapter.cpp "simulation;unit;mujoco")
    add_simulation_test(ut_contact_manager
        test/unit/UT-SIM-MOD-006_contact_manager.cpp "simulation;unit;mujoco")
    add_simulation_test(it_sim_single_joint
        test/integration/IT-SIM_single_joint.cpp "simulation;integration;mujoco")
    add_simulation_test(it_sim_actuator_plant
        test/integration/IT-SIM_actuator_plant.cpp "simulation;integration;mujoco")
    add_simulation_test(it_sim_control_actuator_plant
        test/integration/IT-SIM_control_actuator_plant.cpp "simulation;integration;mujoco")
endif()
