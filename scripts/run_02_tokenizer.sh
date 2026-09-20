#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_JAWIKI_DIR/train.jsonl"
agent_require_file "$AI_CPP_CONVERSATION_DIR/train.jsonl"
agent_require_memory
if [ -f "$AI_CPP_TOKENIZER" ]; then
    echo "tokenizer already exists; keeping it: $AI_CPP_TOKENIZER"
    exit 0
fi
mkdir -p "$(dirname -- "$AI_CPP_TOKENIZER")"
exec "$agent_model_cli" tokenizer-balanced \
    --jawiki-train "$AI_CPP_JAWIKI_DIR/train.jsonl" \
    --conversation-train "$AI_CPP_CONVERSATION_DIR/train.jsonl" \
    --output "$AI_CPP_TOKENIZER" --vocabulary 8192 \
    --jawiki-bytes 234881024 --conversation-bytes 33554432 \
    --seed "$AI_CPP_SEED"
