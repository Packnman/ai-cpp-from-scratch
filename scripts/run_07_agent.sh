#!/bin/sh
set -eu
. "$(dirname -- "$0")/agent_pipeline_env.sh"

agent_require_file "$AI_CPP_BUILD_DIR/agent_cli"
agent_require_file "$AI_CPP_SFT_MODEL/manifest.json"
exec "$AI_CPP_BUILD_DIR/agent_cli" --backend hybrid \
    --model "$AI_CPP_SFT_MODEL" --file-root "$agent_repo_root" \
    --memory-db "$AI_CPP_SFT_MODEL/agent-memory.sqlite3" \
    --seed "$AI_CPP_SEED" --temperature 0.8 --top-p 0.9
