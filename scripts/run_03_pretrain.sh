#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_TOKENIZER"
agent_require_file "$AI_CPP_JAWIKI_SHARD_DIR/manifest.json"
agent_require_file "$AI_CPP_JAWIKI_DIR/validation.jsonl"
if agent_is_complete_pretraining "$AI_CPP_PRETRAIN_MODEL"; then
    echo "pretraining already complete; keeping it: $AI_CPP_PRETRAIN_MODEL"
    exit 0
fi
agent_require_memory
# pretrain-sharded resumes automatically only when checkpoint/latest.json exists.
exec "$agent_model_cli" pretrain-sharded \
    --shard-manifest "$AI_CPP_JAWIKI_SHARD_DIR/manifest.json" \
    --validation "$AI_CPP_JAWIKI_DIR/validation.jsonl" \
    --tokenizer "$AI_CPP_TOKENIZER" --output "$AI_CPP_PRETRAIN_MODEL" \
    --layers 6 --embedding 320 --heads 5 --feed-forward 1280 \
    --context 1024 --dropout 0.1 --epochs 1 --max-shards "$AI_CPP_MAX_SHARDS" \
    --validation-token-limit "$AI_CPP_VALIDATION_TOKENS" \
    --batch-size "$AI_CPP_BATCH_SIZE" --accumulation "$AI_CPP_ACCUMULATION" \
    --learning-rate "$AI_CPP_LEARNING_RATE" \
    --minimum-learning-rate "$AI_CPP_MINIMUM_LEARNING_RATE" \
    --warmup-updates "$AI_CPP_WARMUP_UPDATES" \
    --decay-updates "$AI_CPP_DECAY_UPDATES" \
    --max-device-bytes "$AI_CPP_MAX_DEVICE_BYTES" --seed "$AI_CPP_SEED"
