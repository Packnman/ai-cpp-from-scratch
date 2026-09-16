#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_JAWIKI_DIR/train.jsonl"
exec "$agent_model_cli" shard-jawiki \
    --source "$AI_CPP_JAWIKI_DIR/train.jsonl" \
    --output "$AI_CPP_JAWIKI_SHARD_DIR" \
    --shard-bytes "$AI_CPP_JAWIKI_SHARD_BYTES"
