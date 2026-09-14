#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_PRETRAIN_MODEL/manifest.json"
agent_require_file "$AI_CPP_SFT_DATA_DIR/train.jsonl"
agent_require_file "$AI_CPP_SFT_DATA_DIR/validation.jsonl"
agent_require_memory
agent_refuse_checkpoint "$AI_CPP_SFT_MODEL"
exec "$agent_model_cli" sft \
    --train "$AI_CPP_SFT_DATA_DIR/train.jsonl" \
    --validation "$AI_CPP_SFT_DATA_DIR/validation.jsonl" \
    --source "$AI_CPP_PRETRAIN_MODEL" --output "$AI_CPP_SFT_MODEL" \
    --steps "$AI_CPP_SFT_STEPS" --batch-size 1 \
    --accumulation "$AI_CPP_SFT_ACCUMULATION" --seed "$AI_CPP_SEED"
