#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

cmake -S "$agent_repo_root" -B "$AI_CPP_BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DAI_CPP_BUILD_MODEL=ON -DBUILD_TESTING=ON
cmake --build "$AI_CPP_BUILD_DIR" -j"$AI_CPP_JOBS"
ctest --test-dir "$AI_CPP_BUILD_DIR" --output-on-failure
