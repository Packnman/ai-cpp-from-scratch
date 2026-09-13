#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 設定: ここを書き換えるか、同名の環境変数で上書きしてください。
# 相対パスはリポジトリルートを基準にします。
BUILD_DIR="${BUILD_DIR:-build/release}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
MEMORY_POOL="${MEMORY_POOL:-1}"
export AI_CPP_CUDA_MEMORY_POOL="${MEMORY_POOL}"
BUILD_JOBS="${BUILD_JOBS:-2}"
RUN_BUILD="${RUN_BUILD:-1}"       # 1: 実行前にビルド、0: ビルド済みを使用
DRY_RUN="${DRY_RUN:-0}"           # 1: コマンド表示のみ（ビルド・学習しない）
TEXT_DATA_DIR="${TEXT_DATA_DIR:-data/jawiki-20260901}"
SAMPLED_TEXT_DATA_DIR="${SAMPLED_TEXT_DATA_DIR:-experiments/jawiki-pretrain-sample}"
TEXT_TRAIN_DOCUMENTS="${TEXT_TRAIN_DOCUMENTS:-100000}"
TEXT_VALIDATION_DOCUMENTS="${TEXT_VALIDATION_DOCUMENTS:-2000}"
TEXT_TEST_DOCUMENTS="${TEXT_TEST_DOCUMENTS:-2000}"
TEXT_TRAIN_EVERY="${TEXT_TRAIN_EVERY:-12}"
TEXT_VALIDATION_EVERY="${TEXT_VALIDATION_EVERY:-32}"
TEXT_TEST_EVERY="${TEXT_TEST_EVERY:-33}"
DATA_DIR="${DATA_DIR:-data/conversation}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-models/jawiki_pretrained}"
OUTPUT_DIR="${OUTPUT_DIR:-models/conversation_bpe_finetuned}"
# 空: jawiki事前学習から実行、指定あり: そのモデルから会話追加学習だけを実行
FROM_MODEL="${FROM_MODEL-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE="${ACCUMULATE:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
CLIP_NORM="${CLIP_NORM:-1.0}"
SEED="${SEED:-42}"
MAX_BATCHES="${MAX_BATCHES:-0}"   # 0: 全件、正数: 各 split/epoch のバッチ数上限
PRETRAIN_TOKEN_BUDGET="${PRETRAIN_TOKEN_BUDGET:-15000000}"
FINETUNE_TOKEN_BUDGET="${FINETUNE_TOKEN_BUDGET:-15000000}"

# 新規学習専用。追加学習ではこれらを渡さず、保存モデルの設定を使用します。
BLOCKS="${BLOCKS:-4}"
EMBEDDING="${EMBEDDING:-256}"
HEADS="${HEADS:-4}"
HIDDEN="${HIDDEN:-1024}"
CONTEXT="${CONTEXT:-512}"
DROPOUT="${DROPOUT:-0.1}"
TOKENIZER="${TOKENIZER:-bpe}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-experiments/quality/common-tokenizer.model}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"

cd "${repo_root}"
source_model="${FROM_MODEL:-${PRETRAIN_OUTPUT_DIR}}"
pretrain_args=()
if [[ -z "${FROM_MODEL}" ]]; then
    pretrain_args=("${BUILD_DIR}/main_train" "${SAMPLED_TEXT_DATA_DIR}" "${PRETRAIN_OUTPUT_DIR}"
        --epochs "${PRETRAIN_EPOCHS}" --batch "${BATCH_SIZE}"
        --accumulate "${ACCUMULATE}" --lr "${LEARNING_RATE}"
        --clip "${CLIP_NORM}" --seed "${SEED}" --max-batches "${MAX_BATCHES}"
        --token-budget "${PRETRAIN_TOKEN_BUDGET}"
        --blocks "${BLOCKS}" --embedding "${EMBEDDING}" --heads "${HEADS}"
        --hidden "${HIDDEN}" --context "${CONTEXT}" --dropout "${DROPOUT}"
        --tokenizer "${TOKENIZER}" --vocab-size "${VOCAB_SIZE}"
        --data-format text --loss-target all)
    if [[ -n "${TOKENIZER_MODEL}" ]]; then
        pretrain_args+=(--tokenizer-model "${TOKENIZER_MODEL}")
    fi
fi

finetune_args=("${BUILD_DIR}/main_train" "${DATA_DIR}" "${OUTPUT_DIR}"
    --from-model "${source_model}" --epochs "${FINETUNE_EPOCHS}"
    --batch "${BATCH_SIZE}" --accumulate "${ACCUMULATE}"
    --lr "${LEARNING_RATE}" --clip "${CLIP_NORM}" --seed "${SEED}"
    --max-batches "${MAX_BATCHES}" --token-budget "${FINETUNE_TOKEN_BUDGET}"
    --data-format conversation --loss-target response)

print_command() {
    printf 'AI_CPP_CUDA_MEMORY_POOL=%s Command: ' "${MEMORY_POOL}"
    printf '%q ' "$@"
    printf '\n'
}

sample_text_split() {
    local split="$1"
    local every="$2"
    local maximum="$3"
    local input="${TEXT_DATA_DIR}/${split}.jsonl"
    local output="${SAMPLED_TEXT_DATA_DIR}/${split}.jsonl"
    if [[ ! "${every}" =~ ^[1-9][0-9]*$ || ! "${maximum}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Sampling interval and document count must be positive integers: ${split}" >&2
        exit 1
    fi
    echo "Sampling jawiki ${split}: every ${every} document(s), up to ${maximum}"
    awk -v every="${every}" -v maximum="${maximum}" '
        (NR - 1) % every == 0 {
            print
            if (++selected >= maximum) exit
        }
    ' "${input}" > "${output}"
    local selected
    selected="$(wc -l < "${output}")"
    if [[ "${selected}" -eq 0 ]]; then
        echo "Sampling produced no documents from ${input}." >&2
        exit 1
    fi
    echo "Sampled ${selected} documents: ${output}"
}

if [[ "${#pretrain_args[@]}" -gt 0 ]]; then
    print_command "${pretrain_args[@]}"
fi
print_command "${finetune_args[@]}"
if [[ "${DRY_RUN}" == 1 ]]; then
    exit 0
fi
if [[ "${#pretrain_args[@]}" -gt 0 ]]; then
    mkdir -p "${SAMPLED_TEXT_DATA_DIR}"
    sample_text_split train "${TEXT_TRAIN_EVERY}" "${TEXT_TRAIN_DOCUMENTS}"
    sample_text_split validation "${TEXT_VALIDATION_EVERY}" "${TEXT_VALIDATION_DOCUMENTS}"
    sample_text_split test "${TEXT_TEST_EVERY}" "${TEXT_TEST_DOCUMENTS}"
fi
if [[ "${RUN_BUILD}" == 1 ]]; then
    cmake -S "${repo_root}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    cmake --build "${BUILD_DIR}" --target main_train -j "${BUILD_JOBS}"
fi
if [[ "${#pretrain_args[@]}" -gt 0 ]]; then
    "${pretrain_args[@]}"
fi
exec "${finetune_args[@]}"
