#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_SFT_MODEL/checkpoint/latest.json"
agent_require_file "$AI_CPP_SFT_DATA_DIR/train.jsonl"
agent_require_file "$AI_CPP_SFT_DATA_DIR/validation.jsonl"
agent_require_memory
exec "$agent_model_cli" resume --kind sft \
    --train "$AI_CPP_SFT_DATA_DIR/train.jsonl" \
    --validation "$AI_CPP_SFT_DATA_DIR/validation.jsonl" \
    --model "$AI_CPP_SFT_MODEL" --steps "$AI_CPP_RESUME_STEPS" \
    --batch-size 1 --accumulation "$AI_CPP_SFT_ACCUMULATION" \
    --seed "$AI_CPP_SEED"
