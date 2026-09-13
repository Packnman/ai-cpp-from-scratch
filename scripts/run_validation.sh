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
DRY_RUN="${DRY_RUN:-0}"           # 1: コマンド表示のみ（ビルド・評価しない）
MODE="${MODE:-chat}"         # evaluate: loss/perplexity 評価、chat: 対話生成
MODEL_DIR="${MODEL_DIR:-models/conversation}"

# evaluate 専用
DATA_DIR="${DATA_DIR:-data/conversation}"
SPLIT="${SPLIT:-validation}"     # validation または test
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_BATCHES="${MAX_BATCHES:-0}"  # 0: 全件、正数: バッチ数上限

# chat 専用
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-40}"
MAX_TOKENS="${MAX_TOKENS:-256}"
INPUT_CONTEXT="${INPUT_CONTEXT:-0}" # 0: 保存モデルの文脈長
SEED="${SEED:-42}"

cd "${repo_root}"
case "${MODE}" in
    evaluate)
        args=("${BUILD_DIR}/main_validation" "${DATA_DIR}" "${MODEL_DIR}"
            --split "${SPLIT}" --batch "${BATCH_SIZE}" --max-batches "${MAX_BATCHES}")
        ;;
    chat)
        args=("${BUILD_DIR}/main_validation" chat "${MODEL_DIR}"
            --temperature "${TEMPERATURE}" --top-k "${TOP_K}"
            --max-tokens "${MAX_TOKENS}" --input-context "${INPUT_CONTEXT}" --seed "${SEED}")
        ;;
    *)
        echo "MODE must be evaluate or chat: ${MODE}" >&2
        exit 1
        ;;
esac

# 評価結果の JSON をリダイレクトできるよう、実行情報は stderr へ出します。
printf 'AI_CPP_CUDA_MEMORY_POOL=%s Command: ' "${MEMORY_POOL}" >&2
printf '%q ' "${args[@]}" >&2
printf '\n' >&2
if [[ "${DRY_RUN}" == 1 ]]; then
    exit 0
fi
if [[ "${RUN_BUILD}" == 1 ]]; then
    cmake -S "${repo_root}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" >&2
    cmake --build "${BUILD_DIR}" --target main_validation -j "${BUILD_JOBS}" >&2
fi
exec "${args[@]}"
