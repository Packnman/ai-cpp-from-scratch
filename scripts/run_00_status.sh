#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_status_file() {
    if [ -f "$2" ]; then
        echo "[done]    $1"
    else
        echo "[pending] $1"
    fi
}

agent_status_file "model build" "$agent_model_cli"
agent_status_file "v3 balanced tokenizer" "$AI_CPP_TOKENIZER"
agent_status_file "Jawiki shard manifest" "$AI_CPP_JAWIKI_SHARD_DIR/manifest.json"
if agent_is_complete_pretraining "$AI_CPP_PRETRAIN_MODEL"; then
    echo "[done]    v3 Jawiki pretraining"
else
    echo "[pending] v3 Jawiki pretraining"
fi
agent_status_file "$AI_CPP_SFT_PROFILE SFT train data" "$AI_CPP_SFT_DATA_DIR/train.jsonl"
agent_status_file "$AI_CPP_SFT_PROFILE SFT validation data" "$AI_CPP_SFT_DATA_DIR/validation.jsonl"
agent_status_file "$AI_CPP_SFT_PROFILE SFT bundle" "$AI_CPP_SFT_MODEL/manifest.json"

if [ ! -x "$agent_model_cli" ]; then
    echo "next: ./scripts/run_01_build.sh"
elif [ ! -f "$AI_CPP_TOKENIZER" ]; then
    echo "next: ./scripts/run_02_tokenizer.sh"
elif [ ! -f "$AI_CPP_JAWIKI_SHARD_DIR/manifest.json" ]; then
    echo "next: ./scripts/run_03_shard_jawiki.sh"
elif ! agent_is_complete_pretraining "$AI_CPP_PRETRAIN_MODEL"; then
    echo "next: ./scripts/run_03_pretrain.sh"
elif [ ! -f "$AI_CPP_SFT_DATA_DIR/train.jsonl" ] ||
     [ ! -f "$AI_CPP_SFT_DATA_DIR/validation.jsonl" ]; then
    echo "next: ./scripts/run_04_generate_sft.sh"
elif [ ! -f "$AI_CPP_SFT_MODEL/manifest.json" ]; then
    echo "next: ./scripts/run_05_sft.sh"
else
    echo "next: ./scripts/run_06_validate.sh, then ./scripts/run_07_agent.sh"
fi
