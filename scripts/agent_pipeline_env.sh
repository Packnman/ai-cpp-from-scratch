#!/bin/sh
# Shared defaults for run_01_build.sh through run_07_agent.sh.

agent_script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
agent_repo_root=$(CDPATH= cd -- "$agent_script_dir/.." && pwd)

AI_CPP_BUILD_DIR=${AI_CPP_BUILD_DIR:-"$agent_repo_root/build/model"}
AI_CPP_JOBS=${AI_CPP_JOBS:-2}
AI_CPP_JAWIKI_DIR=${AI_CPP_JAWIKI_DIR:-"$agent_repo_root/data/jawiki-20260901"}
AI_CPP_CONVERSATION_DIR=${AI_CPP_CONVERSATION_DIR:-"$agent_repo_root/data/conversation"}
AI_CPP_TOKENIZER=${AI_CPP_TOKENIZER:-"$agent_repo_root/models/agent_v1_smoke/tokenizer.model"}
AI_CPP_PRETRAIN_MODEL=${AI_CPP_PRETRAIN_MODEL:-"$agent_repo_root/models/agent_v1_pretrain_15m"}
AI_CPP_JAWIKI_SHARD_DIR=${AI_CPP_JAWIKI_SHARD_DIR:-"$agent_repo_root/data/jawiki-20260901-shards"}
AI_CPP_JAWIKI_SHARD_BYTES=${AI_CPP_JAWIKI_SHARD_BYTES:-134217728}
AI_CPP_FULL_PRETRAIN_MODEL=${AI_CPP_FULL_PRETRAIN_MODEL:-"$agent_repo_root/models/agent_v2_pretrain_jawiki_full"}
AI_CPP_MAX_SHARDS=${AI_CPP_MAX_SHARDS:-36}

# Next-generation Jawiki model; intentionally separate from the v2 baseline.
AI_CPP_NEXT_TOKENIZER=${AI_CPP_NEXT_TOKENIZER:-"$agent_repo_root/models/agent_v3/tokenizer.model"}
AI_CPP_NEXT_PRETRAIN_MODEL=${AI_CPP_NEXT_PRETRAIN_MODEL:-"$agent_repo_root/models/agent_v3_pretrain_jawiki"}
AI_CPP_NEXT_JAWIKI_BYTES=${AI_CPP_NEXT_JAWIKI_BYTES:-234881024}
AI_CPP_NEXT_CONVERSATION_BYTES=${AI_CPP_NEXT_CONVERSATION_BYTES:-33554432}
AI_CPP_NEXT_BATCH_SIZE=${AI_CPP_NEXT_BATCH_SIZE:-2}
AI_CPP_NEXT_ACCUMULATION=${AI_CPP_NEXT_ACCUMULATION:-4}
AI_CPP_NEXT_WARMUP_UPDATES=${AI_CPP_NEXT_WARMUP_UPDATES:-5000}
AI_CPP_NEXT_DECAY_UPDATES=${AI_CPP_NEXT_DECAY_UPDATES:-225000}
AI_CPP_NEXT_LEARNING_RATE=${AI_CPP_NEXT_LEARNING_RATE:-0.0003}
AI_CPP_NEXT_MINIMUM_LEARNING_RATE=${AI_CPP_NEXT_MINIMUM_LEARNING_RATE:-0.00003}
AI_CPP_NEXT_MAX_DEVICE_BYTES=${AI_CPP_NEXT_MAX_DEVICE_BYTES:-8053063680}
AI_CPP_SFT_SOURCE_MODEL=${AI_CPP_SFT_SOURCE_MODEL:-"$AI_CPP_FULL_PRETRAIN_MODEL"}
AI_CPP_SFT_PROFILE=${AI_CPP_SFT_PROFILE:-chat}
if [ "$AI_CPP_SFT_PROFILE" = agent ]; then
    agent_default_sft_data="$agent_repo_root/data/agent-v2-sft"
    agent_default_sft_model="$agent_repo_root/models/agent_v2_sft"
else
    agent_default_sft_data="$agent_repo_root/data/realpersona-chat-sft"
    agent_default_sft_model="$agent_repo_root/models/agent_v2_chat_sft"
fi
AI_CPP_SFT_DATA_DIR=${AI_CPP_SFT_DATA_DIR:-"$agent_default_sft_data"}
AI_CPP_SFT_MODEL=${AI_CPP_SFT_MODEL:-"$agent_default_sft_model"}

AI_CPP_PRETRAIN_STEPS=${AI_CPP_PRETRAIN_STEPS:-100000}
AI_CPP_PRETRAIN_TOKENS=${AI_CPP_PRETRAIN_TOKENS:-15000000}
AI_CPP_VALIDATION_TOKENS=${AI_CPP_VALIDATION_TOKENS:-260000}
AI_CPP_PRETRAIN_ACCUMULATION=${AI_CPP_PRETRAIN_ACCUMULATION:-2}
AI_CPP_JMULTIWOZ_DIR=${AI_CPP_JMULTIWOZ_DIR:-}
AI_CPP_JMULTIWOZ_REVISION=${AI_CPP_JMULTIWOZ_REVISION:-}
if [ "$AI_CPP_SFT_PROFILE" = agent ]; then
    agent_default_sft_steps=50000
    agent_default_autoregressive=true
else
    agent_default_sft_steps=5000
    agent_default_autoregressive=false
fi
AI_CPP_SFT_STEPS=${AI_CPP_SFT_STEPS:-"$agent_default_sft_steps"}
AI_CPP_AUTOREGRESSIVE_VALIDATION=${AI_CPP_AUTOREGRESSIVE_VALIDATION:-"$agent_default_autoregressive"}
AI_CPP_SFT_ACCUMULATION=${AI_CPP_SFT_ACCUMULATION:-4}
AI_CPP_RESUME_STEPS=${AI_CPP_RESUME_STEPS:-1000}
AI_CPP_SEED=${AI_CPP_SEED:-42}
AI_CPP_MIN_AVAILABLE_KIB=${AI_CPP_MIN_AVAILABLE_KIB:-4194304}

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
        echo "checkpoint already exists in $1; use the matching resume script" >&2
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
