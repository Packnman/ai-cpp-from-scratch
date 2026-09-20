#!/bin/sh
set -eu

brain_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
brain_repo_root=$(CDPATH= cd -- "$brain_script_dir/../.." && pwd)
brain_mode=${1:-all}
if [ "$#" -gt 0 ]; then
    shift
fi

case "$brain_mode" in
spec)
    brain_default_build="$brain_repo_root/build/brain-tests"
    brain_label=spec
    brain_sanitizers=OFF
    ;;
all)
    brain_default_build="$brain_repo_root/build/brain-tests"
    brain_label=brain
    brain_sanitizers=OFF
    ;;
sanitizer)
    brain_default_build="$brain_repo_root/build/brain-tests-sanitized"
    brain_label=brain
    brain_sanitizers=ON
    ;;
*)
    echo "usage: $0 [spec|all|sanitizer] [ctest options...]" >&2
    exit 2
    ;;
esac

brain_build_dir=${AI_CPP_BRAIN_BUILD_DIR:-"$brain_default_build"}
brain_jobs=${AI_CPP_JOBS:-2}
brain_spec_targets="ut_brn_mod_001_input_adapter ut_brn_mod_002_preprocess
ut_brn_mod_003_world_state_manager ut_brn_mod_004_goal_manager
ut_brn_mod_005_constraint_manager ut_brn_mod_006_memory_manager
ut_brn_mod_007_policy_manager ut_brn_mod_008_planner
ut_brn_mod_009_execution_manager ut_brn_mod_010_post_process
ut_brn_mod_011_external_ai_adapter ut_brn_mod_012_log_trace_manager
it_brain_integration brain_context_model_check"
brain_regression_targets="brain_common_check brain_phase2_check
brain_phase3_4_check brain_phase5_8_check"
brain_targets=$brain_spec_targets
if [ "$brain_label" = brain ]; then
    brain_targets="$brain_targets $brain_regression_targets"
fi

cmake -S "$brain_repo_root" -B "$brain_build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
    -DAI_CPP_BUILD_CUDA_LIB=OFF -DAI_CPP_BUILD_MODEL=OFF \
    -DAI_CPP_ENABLE_SANITIZERS="$brain_sanitizers"
# Target names are fixed above; intentional word splitting passes each to CMake.
cmake --build "$brain_build_dir" -j"$brain_jobs" --target $brain_targets
exec ctest --test-dir "$brain_build_dir" -L "$brain_label" \
    --output-on-failure "$@"
