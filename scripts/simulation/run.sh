#!/bin/sh
set -eu

simulation_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
simulation_repo_root=$(CDPATH= cd -- "$simulation_script_dir/../.." && pwd)
simulation_mode=${1:-test}
if [ "$#" -gt 0 ]; then
    shift
fi

simulation_jobs=${AI_CPP_JOBS:-2}
simulation_generator=${AI_CPP_CMAKE_GENERATOR:-Ninja}
simulation_model_dir="$simulation_repo_root/Simulation/model"
simulation_test_targets="actuator_sim_component_check simulation_cli \
ut_simulation_manager ut_mujoco_plant ut_actuator_adapter \
ut_state_adapter ut_contact_manager it_sim_single_joint \
it_sim_actuator_plant it_sim_control_actuator_plant"

simulation_prepare_x11() {
    if [ -n "${AI_CPP_X11_DISPLAY:-}" ]; then
        DISPLAY=$AI_CPP_X11_DISPLAY
    else
        case "${DISPLAY:-}" in
        *:*) ;;
        *) DISPLAY=host.docker.internal:0.0 ;;
        esac
    fi
    export DISPLAY

    # VcXsrv's -wgl path exposes the OpenGL version required by MuJoCo. Forcing
    # indirect GLX limits this setup to OpenGL 1.4 and MuJoCo rejects it.
    LIBGL_ALWAYS_INDIRECT=${LIBGL_ALWAYS_INDIRECT:-0}
    export LIBGL_ALWAYS_INDIRECT

    echo "X11 viewer: DISPLAY=$DISPLAY LIBGL_ALWAYS_INDIRECT=$LIBGL_ALWAYS_INDIRECT"
}

simulation_configure() {
    simulation_build_dir=$1
    simulation_sanitizers=$2
    simulation_viewer=$3
    cmake -S "$simulation_repo_root" -B "$simulation_build_dir" \
        -G "$simulation_generator" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DBUILD_TESTING=ON \
        -DAI_CPP_BUILD_CUDA_LIB=OFF \
        -DAI_CPP_BUILD_SIMULATION=ON \
        -DAI_CPP_FETCH_MUJOCO=ON \
        -DAI_CPP_ENABLE_MUJOCO_VIEWER="$simulation_viewer" \
        -DAI_CPP_FETCH_GLFW=ON \
        -DAI_CPP_ENABLE_SANITIZERS="$simulation_sanitizers"
}

simulation_build_tests() {
    simulation_build_dir=$1
    # Word splitting is intentional: CMake expects each target separately.
    # shellcheck disable=SC2086
    cmake --build "$simulation_build_dir" -j"$simulation_jobs" \
        --target $simulation_test_targets
}

case "$simulation_mode" in
test)
    simulation_build_dir=${AI_CPP_SIMULATION_BUILD_DIR:-"$simulation_repo_root/build/simulation"}
    simulation_configure "$simulation_build_dir" OFF OFF
    simulation_build_tests "$simulation_build_dir"
    exec ctest --test-dir "$simulation_build_dir" -L simulation \
        --output-on-failure "$@"
    ;;
sanitizer)
    simulation_build_dir=${AI_CPP_SIMULATION_BUILD_DIR:-"$simulation_repo_root/build/simulation-sanitized"}
    simulation_configure "$simulation_build_dir" ON OFF
    simulation_build_tests "$simulation_build_dir"
    exec ctest --test-dir "$simulation_build_dir" -L simulation \
        --output-on-failure "$@"
    ;;
headless)
    simulation_build_dir=${AI_CPP_SIMULATION_BUILD_DIR:-"$simulation_repo_root/build/simulation"}
    simulation_configure "$simulation_build_dir" OFF OFF
    cmake --build "$simulation_build_dir" -j"$simulation_jobs" \
        --target simulation_cli
    exec "$simulation_build_dir/Simulation/simulation_cli" \
        --headless --model "$simulation_model_dir/humanoid/robot.xml" \
        --steps "${AI_CPP_SIMULATION_STEPS:-1000}" "$@"
    ;;
viewer)
    simulation_prepare_x11
    simulation_build_dir=${AI_CPP_SIMULATION_BUILD_DIR:-"$simulation_repo_root/build/simulation-viewer"}
    simulation_configure "$simulation_build_dir" OFF ON
    cmake --build "$simulation_build_dir" -j"$simulation_jobs" \
        --target simulation_cli
    exec "$simulation_build_dir/Simulation/simulation_cli" \
        --viewer --model "$simulation_model_dir/humanoid/robot.xml" "$@"
    ;;
humanoid-viewer)
    simulation_prepare_x11
    simulation_build_dir=${AI_CPP_SIMULATION_BUILD_DIR:-"$simulation_repo_root/build/simulation-viewer"}
    simulation_configure "$simulation_build_dir" OFF ON
    cmake --build "$simulation_build_dir" -j"$simulation_jobs" \
        --target humanoid_demo
    exec "$simulation_build_dir/Simulation/humanoid_demo" \
        --viewer "$@"
    ;;
smoke)
    simulation_build_dir=${AI_CPP_SIMULATION_BUILD_DIR:-"$simulation_repo_root/build/simulation"}
    simulation_configure "$simulation_build_dir" OFF OFF
    cmake --build "$simulation_build_dir" -j"$simulation_jobs" \
        --target simulation_cli
    for simulation_model in \
        test/single_joint.xml \
        test/double_pendulum.xml \
        test/simple_leg.xml \
        humanoid/robot.xml
    do
        "$simulation_build_dir/Simulation/simulation_cli" \
            --headless --model "$simulation_model_dir/$simulation_model" \
            --steps "${AI_CPP_SIMULATION_SMOKE_STEPS:-2}"
    done
    ;;
*)
    echo "usage: $0 [test|sanitizer|headless|viewer|humanoid-viewer|smoke] [options...]" >&2
    exit 2
    ;;
esac
