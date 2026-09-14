# ai_cpp Agent MVP

小さな推論器、短いcontext、外部memory、反復実行を組み合わせるC++20製AgentのMVPです。設計の正本は
[`docs/codex_agent_model_context.md`](docs/codex_agent_model_context.md)です。

既定はCPUだけで動く決定論的なrule backendです。任意のmodel buildでは7,624,192 parameterの
decoder-only Transformer、byte-fallback BPE、Jawiki事前学習、9-mode SFT、resumable
checkpoint、model backendを有効化できます。リポジトリには学習済み重量を含めません。

## 構成

```text
InputParser -> Router -> Retriever -> Planner -> Executor -> Evaluator
                                             ^             |
                                             +-- Replanner-+
                   -> Aggregator -> Response -> Memory Manager
```

単純会話はPlannerを迂回します。Tool実行結果は必ずEvaluatorを通ります。Task planは最大32件のDAGで、
重複ID、存在しない依存先、cycleを実行前に拒否します。retryは同じtoolにつき2回、replanは3回が上限です。

主なディレクトリは、`include/ai/agent/`と`src/agent/`（Agent本体）、`app/`（CLI）、`tests/`、
固定版依存を置く`third_party/`、未変更の既存CUDAライブラリ`lib/`です。

## Build / test

通常buildはnetworkを使いません。

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

sanitizer buildは`-DAI_CPP_ENABLE_SANITIZERS=ON`を追加します。モデルbuildは次のとおりです。

```sh
cmake -S . -B build/model -G Ninja -DAI_CPP_BUILD_MODEL=ON -DBUILD_TESTING=ON
cmake --build build/model
ctest --test-dir build/model --output-on-failure
```

## CLI

`agent_cli`は名前付きoptionだけを受け付けます。既定backendはruleです。

```text
/calc (2 + 3) * 4
/read README.md
/remember project AgentではC++20を使う
/recall C++20
/quit
```

```sh
./build/model/agent_cli --backend rule --file-root . --memory-db agent.sqlite3
./build/model/agent_cli --backend model --model models/agent-sft \
  --file-root . --memory-db agent.sqlite3 --seed 42 --temperature 0.8 --top-p 0.9
```

`file.read`はroot以下の通常UTF-8 fileだけを最大1 MiBまで読み、絶対path、path traversal、symlink脱出を拒否します。
`calculator.calculate`は四則演算、単項符号、括弧だけを解釈し、codeを実行しません。

長期memoryはSemantic、Episodic、Projectの3種です。同一type・本文はupsertされ、FTS5 trigramで日本語を検索します。
3 code point未満はparameter bindingされた完全一致・部分一致検索へ切り替わります。WAL、transaction、5秒busy timeout、
schema version 1 migrationを使用します。

`ModelReasoner`は注入した`ILanguageModel`を使い、設計正本にある9個のspecial modeを切り替えます。
構造化出力は必須fieldとnested schemaを検査し、不正時は修正を一度だけ要求します。
各modeは同じ小型decoder-only Transformerへmulti-task SFTします。新bundleは`ai_cpp_agent_model` version 1で、
旧model bundleとの互換性はありません。

## 学習CLI

`agent_model_cli`の全commandはsmokeと本学習で共通です。tokenizerはJawikiのtrain
JSONLだけを指定し、validation/testを渡せないinterfaceにしています。BPE学習は順序を固定し、
入力を16 MiB、1文を64 KiBに制限するstreaming iteratorを使うため、4.5 GBのsplit全体を
memoryへ載せません。

```sh
M=./build/model/agent_model_cli
$M tokenizer --train data/jawiki/train.jsonl --output models/tokenizer.model --vocabulary 8192
$M pretrain --train data/jawiki/train.jsonl --validation data/jawiki/validation.jsonl \
  --tokenizer models/tokenizer.model --output models/agent-pretrain \
  --steps 100000 --token-budget 15000000 --batch-size 2 --accumulation 8
$M generate-sft --conversation-train data/conversation/train.jsonl \
  --output data/agent-sft/train.jsonl --seed 42
$M sft --train data/agent-sft/train.jsonl --validation data/agent-sft/validation.jsonl \
  --source models/agent-pretrain --output models/agent-sft --steps 5000
$M resume --kind sft --train data/agent-sft/train.jsonl \
  --validation data/agent-sft/validation.jsonl --model models/agent-sft --steps 1000
$M validate --kind sft --data data/agent-sft/validation.jsonl \
  --model models/agent-sft --batch-size 2
```

SFTではmode tokenより前のprompt targetをPADにしてlossから除外します。manifestには構造化modeを
80%とする固定mix weights、checkpointにはweight/Adam fingerprint、step、shuffle RNG、
dropout counter、data/tokenizer fingerprintを保存します。`latest.json`はatomic renameで更新します。

### 段階実行スクリプト

WSL2でメモリ使用量を抑えて順番に実行する場合は、次のスクリプトを使います。buildは最大2並列、
学習はbatch 1で、学習開始時にavailable memoryが4 GiB未満なら停止します。

```sh
./scripts/run_00_status.sh
./scripts/run_01_build.sh
./scripts/run_02_tokenizer.sh
./scripts/run_03_pretrain.sh
./scripts/run_04_generate_sft.sh
./scripts/run_05_sft.sh
./scripts/run_06_validate.sh
./scripts/run_07_agent.sh
```

保存済みcheckpointへpretrainを追加する場合は `run_03_resume_pretrain.sh`、SFTを追加する場合は
`run_05_resume_sft.sh` を使います。
既定tokenizerは検証済みの `models/agent_v1_smoke/tokenizer.model` を再利用します。pathやstep数は
`AI_CPP_TOKENIZER`、`AI_CPP_PRETRAIN_MODEL`、`AI_CPP_SFT_MODEL`、
`AI_CPP_PRETRAIN_TOKENS`、`AI_CPP_VALIDATION_TOKENS`、`AI_CPP_SFT_STEPS` などの
環境変数で変更できます。既定ではJawiki indexもtrain 15M tokens、validation 26万tokensまでに
制限し、resume時にも同じfingerprintになるよう同じ制限を適用します。
