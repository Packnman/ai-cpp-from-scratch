#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/debug"

cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Debug
cmake --build "${build_dir}" --target cuda_check
ctest --test-dir "${build_dir}" --output-on-failure -R '^cuda_check$'
