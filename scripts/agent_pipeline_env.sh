#!/bin/sh
# Shared v3 model-pipeline defaults. Override any AI_CPP_* value as needed.

agent_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
agent_repo_root=$(CDPATH= cd -- "$agent_script_dir/.." && pwd)

AI_CPP_BUILD_DIR=${AI_CPP_BUILD_DIR:-"$agent_repo_root/build/model"}
AI_CPP_JOBS=${AI_CPP_JOBS:-2}
AI_CPP_SEED=${AI_CPP_SEED:-42}
AI_CPP_MIN_AVAILABLE_KIB=${AI_CPP_MIN_AVAILABLE_KIB:-4194304}

AI_CPP_JAWIKI_DIR=${AI_CPP_JAWIKI_DIR:-"$agent_repo_root/data/jawiki-20260901"}
AI_CPP_JAWIKI_SHARD_DIR=${AI_CPP_JAWIKI_SHARD_DIR:-"$agent_repo_root/data/jawiki-20260901-shards"}
AI_CPP_JAWIKI_SHARD_BYTES=${AI_CPP_JAWIKI_SHARD_BYTES:-134217728}
AI_CPP_CONVERSATION_DIR=${AI_CPP_CONVERSATION_DIR:-"$agent_repo_root/data/conversation"}

AI_CPP_TOKENIZER=${AI_CPP_TOKENIZER:-"$agent_repo_root/models/agent_v3/tokenizer.model"}
AI_CPP_PRETRAIN_MODEL=${AI_CPP_PRETRAIN_MODEL:-"$agent_repo_root/models/agent_v3_pretrain_jawiki"}
AI_CPP_MAX_SHARDS=${AI_CPP_MAX_SHARDS:-36}
AI_CPP_VALIDATION_TOKENS=${AI_CPP_VALIDATION_TOKENS:-260000}
AI_CPP_BATCH_SIZE=${AI_CPP_BATCH_SIZE:-2}
AI_CPP_ACCUMULATION=${AI_CPP_ACCUMULATION:-4}
AI_CPP_WARMUP_UPDATES=${AI_CPP_WARMUP_UPDATES:-5000}
AI_CPP_DECAY_UPDATES=${AI_CPP_DECAY_UPDATES:-225000}
AI_CPP_LEARNING_RATE=${AI_CPP_LEARNING_RATE:-0.0003}
AI_CPP_MINIMUM_LEARNING_RATE=${AI_CPP_MINIMUM_LEARNING_RATE:-0.00003}
AI_CPP_MAX_DEVICE_BYTES=${AI_CPP_MAX_DEVICE_BYTES:-8053063680}

AI_CPP_SFT_PROFILE=${AI_CPP_SFT_PROFILE:-chat}
if [ "$AI_CPP_SFT_PROFILE" = agent ]; then
    agent_default_sft_data="$agent_repo_root/data/agent-v3-base-sft"
    agent_default_sft_model="$agent_repo_root/models/agent_v3_agent_sft"
    agent_default_sft_steps=50000
    agent_default_autoregressive=true
elif [ "$AI_CPP_SFT_PROFILE" = chat ]; then
    agent_default_sft_data="$agent_repo_root/data/realpersona-chat-sft"
    agent_default_sft_model="$agent_repo_root/models/agent_v3_chat_sft"
    agent_default_sft_steps=5000
    agent_default_autoregressive=false
else
    echo "AI_CPP_SFT_PROFILE must be chat or agent" >&2
    exit 1
fi
AI_CPP_SFT_DATA_DIR=${AI_CPP_SFT_DATA_DIR:-"$agent_default_sft_data"}
AI_CPP_SFT_MODEL=${AI_CPP_SFT_MODEL:-"$agent_default_sft_model"}
AI_CPP_SFT_STEPS=${AI_CPP_SFT_STEPS:-"$agent_default_sft_steps"}
AI_CPP_SFT_ACCUMULATION=${AI_CPP_SFT_ACCUMULATION:-4}
AI_CPP_AUTOREGRESSIVE_VALIDATION=${AI_CPP_AUTOREGRESSIVE_VALIDATION:-"$agent_default_autoregressive"}
AI_CPP_JMULTIWOZ_DIR=${AI_CPP_JMULTIWOZ_DIR:-}
AI_CPP_JMULTIWOZ_REVISION=${AI_CPP_JMULTIWOZ_REVISION:-}

agent_model_cli="$AI_CPP_BUILD_DIR/agent_model_cli"

agent_require_file() {
    if [ ! -f "$1" ]; then
        echo "missing required file: $1" >&2
        exit 1
    fi
}

agent_require_cli() {
    if [ ! -x "$agent_model_cli" ]; then
        echo "missing $agent_model_cli; run scripts/run_01_build.sh first" >&2
        exit 1
    fi
}

agent_require_memory() {
    agent_available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [ -z "$agent_available_kib" ] ||
       [ "$agent_available_kib" -lt "$AI_CPP_MIN_AVAILABLE_KIB" ]; then
        echo "refusing to start: less than $AI_CPP_MIN_AVAILABLE_KIB KiB memory is available" >&2
        exit 1
    fi
    echo "memory check: $agent_available_kib KiB available"
}

agent_refuse_checkpoint() {
    if [ -f "$1/checkpoint/latest.json" ]; then
        echo "checkpoint already exists in $1; choose a new output directory" >&2
        exit 1
    fi
}

agent_require_complete_pretraining() {
    agent_require_file "$1/manifest.json"
    if ! grep -q '"pretraining_complete": true' "$1/manifest.json"; then
        echo "pretraining is not complete: $1" >&2
        exit 1
    fi
}

agent_is_complete_pretraining() {
    [ -f "$1/manifest.json" ] &&
        grep -q '"pretraining_complete": true' "$1/manifest.json"
}
