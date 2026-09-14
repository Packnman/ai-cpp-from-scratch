#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_CONVERSATION_DIR/train.jsonl"
agent_require_file "$AI_CPP_CONVERSATION_DIR/validation.jsonl"
mkdir -p "$AI_CPP_SFT_DATA_DIR"
if [ -e "$AI_CPP_SFT_DATA_DIR/train.jsonl" ] ||
   [ -e "$AI_CPP_SFT_DATA_DIR/validation.jsonl" ]; then
    echo "SFT corpus already exists in $AI_CPP_SFT_DATA_DIR; keeping it" >&2
    exit 0
fi
"$agent_model_cli" generate-sft \
    --conversation-train "$AI_CPP_CONVERSATION_DIR/train.jsonl" \
    --output "$AI_CPP_SFT_DATA_DIR/train.jsonl" --seed "$AI_CPP_SEED"
"$agent_model_cli" generate-sft \
    --conversation-train "$AI_CPP_CONVERSATION_DIR/validation.jsonl" \
    --output "$AI_CPP_SFT_DATA_DIR/validation.jsonl" --seed "$AI_CPP_SEED"
