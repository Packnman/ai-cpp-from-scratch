#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build/quality-comparison}"
conversation_data="${CONVERSATION_DATA:-${repo_root}/data/conversation}"
text_data="${TEXT_DATA:?Set TEXT_DATA to the pinned jawiki JSONL directory}"
output_root="${OUTPUT_ROOT:-${repo_root}/experiments/quality-comparison}"
tokenizer_model="${TOKENIZER_MODEL:-${output_root}/common-tokenizer.model}"
current_batch="${CURRENT_MICRO_BATCH:-16}"
large_batch="${LARGE_MICRO_BATCH:-4}"
current_accumulate="${CURRENT_ACCUMULATE:-1}"
large_accumulate="${LARGE_ACCUMULATE:-4}"
jobs="${BUILD_JOBS:-2}"

mkdir -p "${output_root}"
cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --target main_train main_validation -j "${jobs}"

if [[ ! -f "${tokenizer_model}" ]]; then
    "${build_dir}/main_train" tokenizer \
        "${conversation_data}/train.jsonl" "${text_data}/train.jsonl" \
        "${tokenizer_model}" --vocab-size 4096
fi

common=(--tokenizer bpe --tokenizer-model "${tokenizer_model}" --context 512
        --epochs 100000 --seed 42)
current=(--blocks 4 --embedding 256 --heads 4 --hidden 1024
         --batch "${current_batch}" --accumulate "${current_accumulate}")
large=(--blocks 6 --embedding 384 --heads 6 --hidden 1536
       --batch "${large_batch}" --accumulate "${large_accumulate}")

train_new() {
    local data="$1" output="$2" budget="$3" format="$4"
    shift 4
    [[ ! -e "${output}" ]] || {
        echo "Output already exists: ${output}" >&2
        return 1
    }
    "${build_dir}/main_train" "${data}" "${output}" \
        "${common[@]}" "$@" --data-format "${format}" \
        --loss-target "$([[ "${format}" == conversation ]] && echo response || echo all)" \
        --token-budget "${budget}"
}

finetune_conversation() {
    local source="$1" output="$2"
    [[ ! -e "${output}" ]] || {
        echo "Output already exists: ${output}" >&2
        return 1
    }
    "${build_dir}/main_train" "${conversation_data}" "${output}" \
        --from-model "${source}" --data-format conversation \
        --loss-target response --epochs 100000 --token-budget 15000000 \
        --batch "${current_batch}" --accumulate "${current_accumulate}" --seed 42
}

if [[ "${RUN_D:-0}" == 1 ]]; then
    train_new "${text_data}" "${output_root}/D-pretrain" \
        15000000 text "${large[@]}"
    "${build_dir}/main_train" "${conversation_data}" "${output_root}/D-large-mixed" \
        --from-model "${output_root}/D-pretrain" --data-format conversation \
        --loss-target response --epochs 100000 --token-budget 15000000 \
        --batch "${large_batch}" --accumulate "${large_accumulate}" --seed 42
else
    train_new "${conversation_data}" "${output_root}/A-current-conversation" \
        30000000 conversation "${current[@]}"
    train_new "${text_data}" "${output_root}/B-pretrain" \
        15000000 text "${current[@]}"
    finetune_conversation "${output_root}/B-pretrain" "${output_root}/B-current-mixed"
    train_new "${conversation_data}" "${output_root}/C-large-conversation" \
        30000000 conversation "${large[@]}"
fi

for name in A-current-conversation B-current-mixed C-large-conversation D-large-mixed; do
    model="${output_root}/${name}"
    [[ -d "${model}" ]] || continue
    "${build_dir}/main_validation" "${conversation_data}" "${model}" \
        --split validation --batch 16 --max-batches 0 --loss-target response \
        > "${output_root}/${name}-validation.json"
    BUILD_DIR="${build_dir}" MODEL_DIR="${model}" RUN_BUILD=0 \
        OUTPUT_DIR="${output_root}/${name}-fixed-chat" \
        "${repo_root}/scripts/evaluate_fixed_chat.sh"
done

"${repo_root}/scripts/summarize_quality.py" "${output_root}"
