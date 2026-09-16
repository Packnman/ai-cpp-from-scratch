#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_TOKENIZER"
agent_require_file "$AI_CPP_NEXT_TOKENIZER"
for split in jawiki conversation; do
    if [ "$split" = jawiki ]; then data="$AI_CPP_JAWIKI_DIR/validation.jsonl"; else data="$AI_CPP_CONVERSATION_DIR/validation.jsonl"; fi
    agent_require_file "$data"
    echo "baseline $split"
    "$agent_model_cli" tokenizer-evaluate --tokenizer "$AI_CPP_TOKENIZER" --data "$data" --context 1024
    echo "next $split"
    "$agent_model_cli" tokenizer-evaluate --tokenizer "$AI_CPP_NEXT_TOKENIZER" --data "$data" --context 1024
done
