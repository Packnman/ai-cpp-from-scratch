# CMake generated Testfile for 
# Source directory: /workspaces/ai_cpp/Simulation
# Build directory: /workspaces/ai_cpp/build-viewer/Simulation
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ut_simulation_manager "/workspaces/ai_cpp/build-viewer/Simulation/ut_simulation_manager")
set_tests_properties(ut_simulation_manager PROPERTIES  LABELS "simulation;unit" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;78;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(ut_mujoco_plant "/workspaces/ai_cpp/build-viewer/Simulation/ut_mujoco_plant")
set_tests_properties(ut_mujoco_plant PROPERTIES  LABELS "simulation;unit;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;80;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(ut_actuator_adapter "/workspaces/ai_cpp/build-viewer/Simulation/ut_actuator_adapter")
set_tests_properties(ut_actuator_adapter PROPERTIES  LABELS "simulation;unit;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;82;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(ut_state_adapter "/workspaces/ai_cpp/build-viewer/Simulation/ut_state_adapter")
set_tests_properties(ut_state_adapter PROPERTIES  LABELS "simulation;unit;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;84;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(ut_contact_manager "/workspaces/ai_cpp/build-viewer/Simulation/ut_contact_manager")
set_tests_properties(ut_contact_manager PROPERTIES  LABELS "simulation;unit;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;86;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(it_sim_single_joint "/workspaces/ai_cpp/build-viewer/Simulation/it_sim_single_joint")
set_tests_properties(it_sim_single_joint PROPERTIES  LABELS "simulation;integration;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;88;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(it_sim_actuator_plant "/workspaces/ai_cpp/build-viewer/Simulation/it_sim_actuator_plant")
set_tests_properties(it_sim_actuator_plant PROPERTIES  LABELS "simulation;integration;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;90;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(it_sim_control_actuator_plant "/workspaces/ai_cpp/build-viewer/Simulation/it_sim_control_actuator_plant")
set_tests_properties(it_sim_control_actuator_plant PROPERTIES  LABELS "simulation;integration;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;74;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;92;add_simulation_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
add_test(it_sim_brain_humanoid_demo "/workspaces/ai_cpp/build-viewer/Simulation/it_sim_brain_humanoid_demo")
set_tests_properties(it_sim_brain_humanoid_demo PROPERTIES  LABELS "simulation;brain;integration;mujoco" _BACKTRACE_TRIPLES "/workspaces/ai_cpp/Simulation/CMakeLists.txt;100;add_test;/workspaces/ai_cpp/Simulation/CMakeLists.txt;0;")
subdirs("../_deps/glfw-build")
