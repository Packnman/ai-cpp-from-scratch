#!/usr/bin/env bash
set -euo pipefail
# Supply a stable bundle snapshot, never the directory an active trainer writes.
if [[ $# != 2 ]]; then
    echo "Usage: bash scripts/run_memory_benchmark.sh BUNDLE TRAIN_JSONL" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${BUILD_DIR:-${repo_root}/build/release}/cuda_memory_benchmark"
for repetition in 1 2 3; do
    for mode in 0 1; do
        active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
        if [[ -n "${active//[[:space:]]/}" ]]; then
            echo "GPU compute processes are active; benchmark deferred (PIDs: ${active})." >&2
            exit 1
        fi
        echo "repetition=${repetition} AI_CPP_CUDA_MEMORY_POOL=${mode}" >&2
        AI_CPP_CUDA_MEMORY_POOL="${mode}" "${benchmark}" "$1" "$2"
    done
done
