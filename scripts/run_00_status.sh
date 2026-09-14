#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_status() {
    if [ -e "$2" ]; then
        echo "[done]    $1"
    else
        echo "[pending] $1"
    fi
}

agent_status "01 model build" "$agent_model_cli"
agent_status "02 tokenizer" "$AI_CPP_TOKENIZER"
agent_status "03 Jawiki 15M pretrain" \
    "$AI_CPP_PRETRAIN_MODEL/checkpoint/latest.json"
agent_status "04 SFT corpus" "$AI_CPP_SFT_DATA_DIR/validation.jsonl"
agent_status "05 SFT" "$AI_CPP_SFT_MODEL/checkpoint/latest.json"

if [ ! -x "$agent_model_cli" ]; then
    echo "next: ./scripts/run_01_build.sh"
elif [ ! -f "$AI_CPP_TOKENIZER" ]; then
    echo "next: ./scripts/run_02_tokenizer.sh"
elif [ ! -f "$AI_CPP_PRETRAIN_MODEL/checkpoint/latest.json" ]; then
    echo "next: ./scripts/run_03_pretrain.sh"
elif [ ! -f "$AI_CPP_SFT_DATA_DIR/validation.jsonl" ]; then
    echo "next: ./scripts/run_04_generate_sft.sh"
elif [ ! -f "$AI_CPP_SFT_MODEL/checkpoint/latest.json" ]; then
    echo "next: ./scripts/run_05_sft.sh"
else
    echo "next: ./scripts/run_06_validate.sh, then run_07_agent.sh"
fi
