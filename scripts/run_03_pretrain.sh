#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_TOKENIZER"
agent_require_file "$AI_CPP_JAWIKI_DIR/train.jsonl"
agent_require_file "$AI_CPP_JAWIKI_DIR/validation.jsonl"
agent_require_memory
agent_refuse_checkpoint "$AI_CPP_PRETRAIN_MODEL"
exec "$agent_model_cli" pretrain \
    --train "$AI_CPP_JAWIKI_DIR/train.jsonl" \
    --validation "$AI_CPP_JAWIKI_DIR/validation.jsonl" \
    --tokenizer "$AI_CPP_TOKENIZER" --output "$AI_CPP_PRETRAIN_MODEL" \
    --steps "$AI_CPP_PRETRAIN_STEPS" \
    --token-budget "$AI_CPP_PRETRAIN_TOKENS" \
    --train-token-limit "$AI_CPP_PRETRAIN_TOKENS" \
    --validation-token-limit "$AI_CPP_VALIDATION_TOKENS" \
    --batch-size 1 --accumulation "$AI_CPP_PRETRAIN_ACCUMULATION" \
    --seed "$AI_CPP_SEED"
