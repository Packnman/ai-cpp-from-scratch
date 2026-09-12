#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/debug"
cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Debug
cmake --build "${build_dir}"
ctest --test-dir "${build_dir}" --output-on-failure

if [[ $# -gt 0 ]]; then
    command_name="$1"
    shift
    case "${command_name}" in
        train|validation) exec "${build_dir}/main_${command_name}" "$@" ;;
        *) echo "Usage: $0 [train|validation ARGS...]" >&2; exit 1 ;;
    esac
fi
