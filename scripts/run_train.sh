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
DATA_DIR="${DATA_DIR:-data/conversation}"
OUTPUT_DIR="${OUTPUT_DIR:-models/conversation_finetuned}"
FROM_MODEL="${FROM_MODEL:-models/conversation}"     # 空: 新規学習、指定あり: 保存モデルから追加学習
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.003}"
CLIP_NORM="${CLIP_NORM:-1.0}"
SEED="${SEED:-42}"
MAX_BATCHES="${MAX_BATCHES:-0}"   # 0: 全件、正数: 各 split/epoch のバッチ数上限

# 新規学習専用。追加学習ではこれらを渡さず、保存モデルの設定を使用します。
BLOCKS="${BLOCKS:-4}"
EMBEDDING="${EMBEDDING:-256}"
HEADS="${HEADS:-4}"
HIDDEN="${HIDDEN:-1024}"
CONTEXT="${CONTEXT:-256}"
DROPOUT="${DROPOUT:-0.1}"

cd "${repo_root}"
args=("${BUILD_DIR}/main_train" "${DATA_DIR}" "${OUTPUT_DIR}"
    --epochs "${EPOCHS}" --batch "${BATCH_SIZE}" --lr "${LEARNING_RATE}"
    --clip "${CLIP_NORM}" --seed "${SEED}" --max-batches "${MAX_BATCHES}")
if [[ -n "${FROM_MODEL}" ]]; then
    # OUTPUT_DIR は読み込み元とは別の、新規または空ディレクトリにしてください。
    args+=(--from-model "${FROM_MODEL}")
else
    args+=(--blocks "${BLOCKS}" --embedding "${EMBEDDING}" --heads "${HEADS}"
        --hidden "${HIDDEN}" --context "${CONTEXT}" --dropout "${DROPOUT}")
fi

printf 'AI_CPP_CUDA_MEMORY_POOL=%s Command: ' "${MEMORY_POOL}"
printf '%q ' "${args[@]}"
printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then
    exit 0
fi
if [[ "${RUN_BUILD}" == 1 ]]; then
    cmake -S "${repo_root}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    cmake --build "${BUILD_DIR}" --target main_train -j "${BUILD_JOBS}"
fi
exec "${args[@]}"
