#!/bin/bash

set -e

# ==============================
# 設定
# ==============================
OUTPUT="build/test"

SOURCES=(
    src/cuda_check.cu
)

INCLUDE_DIRS=(
    include
)

# ==============================
# buildディレクトリ作成
# ==============================
mkdir -p build

# ==============================
# include オプション生成
# ==============================
INCLUDES=()

for dir in "${INCLUDE_DIRS[@]}"; do
    INCLUDES+=("-I${dir}")
done

# ==============================
# コンパイル
# ==============================
echo "=== CUDA Compile ==="

nvcc \
    -std=c++20 \
    -O3 \
    "${INCLUDES[@]}" \
    "${SOURCES[@]}" \
    -o "$OUTPUT"

# ==============================
# 実行
# ==============================
echo "=== Run ==="

"./$OUTPUT"