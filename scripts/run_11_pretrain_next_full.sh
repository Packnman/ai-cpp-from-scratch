#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_NEXT_TOKENIZER"
agent_require_file "$AI_CPP_JAWIKI_SHARD_DIR/manifest.json"
agent_require_file "$AI_CPP_JAWIKI_DIR/validation.jsonl"
agent_require_memory
exec "$agent_model_cli" pretrain-sharded \
    --shard-manifest "$AI_CPP_JAWIKI_SHARD_DIR/manifest.json" \
    --validation "$AI_CPP_JAWIKI_DIR/validation.jsonl" \
    --tokenizer "$AI_CPP_NEXT_TOKENIZER" --output "$AI_CPP_NEXT_PRETRAIN_MODEL" \
    --layers 6 --embedding 320 --heads 5 --feed-forward 1280 \
    --context 1024 --dropout 0.1 --epochs 1 --max-shards 36 \
    --validation-token-limit "$AI_CPP_VALIDATION_TOKENS" \
    --batch-size "$AI_CPP_NEXT_BATCH_SIZE" --accumulation "$AI_CPP_NEXT_ACCUMULATION" \
    --learning-rate "$AI_CPP_NEXT_LEARNING_RATE" \
    --minimum-learning-rate "$AI_CPP_NEXT_MINIMUM_LEARNING_RATE" \
    --warmup-updates "$AI_CPP_NEXT_WARMUP_UPDATES" \
    --decay-updates "$AI_CPP_NEXT_DECAY_UPDATES" \
    --max-device-bytes "$AI_CPP_NEXT_MAX_DEVICE_BYTES" --seed "$AI_CPP_SEED"
