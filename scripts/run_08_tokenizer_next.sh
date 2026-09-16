#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_JAWIKI_DIR/train.jsonl"
agent_require_file "$AI_CPP_CONVERSATION_DIR/train.jsonl"
agent_require_memory
if [ -e "$AI_CPP_NEXT_TOKENIZER" ] || [ -e "$AI_CPP_NEXT_TOKENIZER.metadata.json" ]; then
    echo "next tokenizer already exists; keeping it: $AI_CPP_NEXT_TOKENIZER"
    exit 0
fi
mkdir -p "$(dirname -- "$AI_CPP_NEXT_TOKENIZER")"
exec "$agent_model_cli" tokenizer-balanced \
    --jawiki-train "$AI_CPP_JAWIKI_DIR/train.jsonl" \
    --conversation-train "$AI_CPP_CONVERSATION_DIR/train.jsonl" \
    --output "$AI_CPP_NEXT_TOKENIZER" --vocabulary 8192 \
    --jawiki-bytes "$AI_CPP_NEXT_JAWIKI_BYTES" \
    --conversation-bytes "$AI_CPP_NEXT_CONVERSATION_BYTES" \
    --seed "$AI_CPP_SEED"
