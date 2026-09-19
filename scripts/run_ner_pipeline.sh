#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
data_root=${AI_CPP_NER_DATA_DIR:-"$repo_root/data/ner"}
build_root=${AI_CPP_NER_BUILD_DIR:-"$repo_root/build/ner"}
bundle=${AI_CPP_NER_BUNDLE:-"$data_root/bundle/ner.json"}
jobs=${AI_CPP_JOBS:-2}

"$repo_root/scripts/fetch_ner_wikipedia.sh" "$data_root/source/ner.json"
python3 "$repo_root/scripts/convert_ner_wikipedia.py" \
    "$data_root/source/ner.json" "$data_root/processed" \
    --seed 20260918 --max-chars 256
cmake -S "$repo_root" -B "$build_root" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$build_root" -j"$jobs"
ctest --test-dir "$build_root" --output-on-failure
mkdir -p "$(dirname -- "$bundle")"
"$build_root/ner_cli" train \
    --train "$data_root/processed/train.jsonl" \
    --validation "$data_root/processed/validation.jsonl" \
    --output "$bundle" --epochs 20 --embedding-dim 16 \
    --hidden-dim 32 --window 3 --learning-rate 0.01 --seed 42
"$build_root/ner_cli" evaluate --bundle "$bundle" \
    --data "$data_root/processed/test.jsonl"

echo "Run Agent with: $build_root/agent_cli --ner hybrid --ner-bundle $bundle"
