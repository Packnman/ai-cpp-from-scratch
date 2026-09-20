#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_cli
agent_require_file "$AI_CPP_CONVERSATION_DIR/train.jsonl"
agent_require_file "$AI_CPP_CONVERSATION_DIR/validation.jsonl"
mkdir -p "$AI_CPP_SFT_DATA_DIR"
if [ -e "$AI_CPP_SFT_DATA_DIR/train.jsonl" ] &&
   [ -e "$AI_CPP_SFT_DATA_DIR/validation.jsonl" ]; then
    echo "SFT corpus already exists in $AI_CPP_SFT_DATA_DIR; keeping it" >&2
    exit 0
fi
if [ -e "$AI_CPP_SFT_DATA_DIR/train.jsonl" ] ||
   [ -e "$AI_CPP_SFT_DATA_DIR/validation.jsonl" ]; then
    echo "incomplete SFT corpus in $AI_CPP_SFT_DATA_DIR; move it aside or finish it explicitly" >&2
    exit 1
fi
if [ "$AI_CPP_SFT_PROFILE" = agent ]; then
    if [ -z "$AI_CPP_JMULTIWOZ_DIR" ] || [ -z "$AI_CPP_JMULTIWOZ_REVISION" ]; then
        echo "agent SFT requires AI_CPP_JMULTIWOZ_DIR and AI_CPP_JMULTIWOZ_REVISION" >&2
        exit 1
    fi
    agent_require_file "$AI_CPP_JMULTIWOZ_DIR/train.json"
    agent_require_file "$AI_CPP_JMULTIWOZ_DIR/validation.json"
    for split in train validation; do
        "$agent_model_cli" generate-sft \
            --conversation-train "$AI_CPP_CONVERSATION_DIR/$split.jsonl" \
            --output "$AI_CPP_SFT_DATA_DIR/$split.base.jsonl" \
            --seed "$AI_CPP_SEED" --profile agent --split "$split"
        python3 "$agent_repo_root/scripts/convert_summary_sft.py" jmultiwoz \
            --input "$AI_CPP_JMULTIWOZ_DIR/$split.json" \
            --output "$AI_CPP_SFT_DATA_DIR/$split.jmultiwoz.jsonl" \
            --revision "$AI_CPP_JMULTIWOZ_REVISION" --split "$split" \
            --seed "$AI_CPP_SEED"
        python3 "$agent_repo_root/scripts/convert_summary_sft.py" mix \
            --base "$AI_CPP_SFT_DATA_DIR/$split.base.jsonl" \
            --jmultiwoz "$AI_CPP_SFT_DATA_DIR/$split.jmultiwoz.jsonl" \
            --output "$AI_CPP_SFT_DATA_DIR/$split.jsonl" --seed "$AI_CPP_SEED"
    done
    exit 0
fi
"$agent_model_cli" generate-sft \
    --conversation-train "$AI_CPP_CONVERSATION_DIR/train.jsonl" \
    --output "$AI_CPP_SFT_DATA_DIR/train.jsonl" --seed "$AI_CPP_SEED" \
    --profile chat --split train
"$agent_model_cli" generate-sft \
    --conversation-train "$AI_CPP_CONVERSATION_DIR/validation.jsonl" \
    --output "$AI_CPP_SFT_DATA_DIR/validation.jsonl" --seed "$AI_CPP_SEED" \
    --profile chat --split validation
