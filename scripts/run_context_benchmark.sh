#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: bash scripts/run_context_benchmark.sh BUNDLE TRAIN_JSONL" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
benchmark="${BUILD_DIR:-${repo_root}/build/release}/cuda_memory_benchmark"
# Each repetition starts a fresh process from the same immutable model.
# stdout is JSONL; keep stderr as the record of unsuccessful batch sizes.
for batch in 32 16 8 4 2 1; do
    success=1
    for repetition in 1 2 3; do
        active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
        if [[ -n "${active//[[:space:]]/}" ]]; then
            echo "GPU compute processes are active; benchmark deferred (PIDs: ${active})." >&2
            exit 1
        fi
        echo "batch=${batch} repetition=${repetition}" >&2
        if AI_CPP_CUDA_MEMORY_POOL=1 "${benchmark}" "$1" "$2" "${batch}"; then
            :
        else
            status=$?
            if [[ "${status}" != 3 ]]; then exit "${status}"; fi
            echo "OOM at batch=${batch}; retrying smaller batch in a fresh process." >&2
            success=0
            break
        fi
    done
    if [[ "${success}" == 1 ]]; then exit 0; fi
done
echo "No batch size fits." >&2
exit 1
