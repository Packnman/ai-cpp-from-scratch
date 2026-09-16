#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_SFT_MODEL/manifest.json"
agent_require_file "$AI_CPP_SFT_DATA_DIR/validation.jsonl"
agent_require_memory
exec "$agent_model_cli" validate --kind sft \
    --data "$AI_CPP_SFT_DATA_DIR/validation.jsonl" \
    --model "$AI_CPP_SFT_MODEL" --batch-size 1 \
    --autoregressive "$AI_CPP_AUTOREGRESSIVE_VALIDATION" --held-out 500
