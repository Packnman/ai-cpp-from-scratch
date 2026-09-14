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
AI_CPP_SFT_DATA_DIR=${AI_CPP_SFT_DATA_DIR:-"$agent_repo_root/data/agent-sft"}
AI_CPP_SFT_MODEL=${AI_CPP_SFT_MODEL:-"$agent_repo_root/models/agent_v1_sft"}

AI_CPP_PRETRAIN_STEPS=${AI_CPP_PRETRAIN_STEPS:-100000}
AI_CPP_PRETRAIN_TOKENS=${AI_CPP_PRETRAIN_TOKENS:-15000000}
AI_CPP_VALIDATION_TOKENS=${AI_CPP_VALIDATION_TOKENS:-260000}
AI_CPP_PRETRAIN_ACCUMULATION=${AI_CPP_PRETRAIN_ACCUMULATION:-8}
AI_CPP_SFT_STEPS=${AI_CPP_SFT_STEPS:-5000}
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
