#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build/release}"
model_dir="${MODEL_DIR:-${repo_root}/models/conversation}"
output_dir="${OUTPUT_DIR:-${repo_root}/evaluation/fixed-chat}"
run_build="${RUN_BUILD:-1}"

if [[ "${run_build}" == 1 ]]; then
    cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${build_dir}" --target main_validation -j "${BUILD_JOBS:-2}"
fi
mkdir -p "${output_dir}"

questions=(
    "こんにちは。今日は何をしていましたか？"
    "どんな仕事ですか？"
    "好きな食べ物は何ですか？理由も教えてください。"
    "私はカレーが好きです。あなたは？"
    "さっき、あなたは何をしていたと言いましたか？"
)

for top_k in 1 40; do
    printf '%s\n' "${questions[@]}" |
        "${build_dir}/main_validation" chat "${model_dir}"             --temperature 0.8 --top-k "${top_k}" --max-tokens 64             --input-context 0 --seed 123 |
        tee "${output_dir}/top-k-${top_k}.log"
    printf '\n' | tee -a "${output_dir}/top-k-${top_k}.log"
done
