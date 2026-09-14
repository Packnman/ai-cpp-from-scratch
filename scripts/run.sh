#!/bin/sh
set -eu
build_dir=${AI_CPP_BUILD_DIR:-build}
cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build "$build_dir"
exec "$build_dir/agent_cli" "$@"
