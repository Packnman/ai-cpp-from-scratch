# ai_cpp Agent MVP

全体構想は [docs/agent_overview.md](docs/agent_overview.md)、クラス図と主要sequenceは [docs/agent_uml.md](docs/agent_uml.md) を参照してください。

設計書、公開ヘッダー別資料、学習・評価文書は [docs/README.md](docs/README.md) から参照できます。

日本語NERの実装・データ・学習・評価手順は [docs/japanese_ner.md](docs/japanese_ner.md) を参照してください。

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

公開APIは`include/`、C++実装は`src/`、実行入口は`app/`と`train/`へ分離しています。
テストは`tests/{agent,brain,model,ner}`、再現用スクリプトは`scripts/`、固定版依存は
`third_party/`です。詳細な配置規則は
[docs/project_structure.md](docs/project_structure.md)を参照してください。

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
./build/model/agent_cli --backend hybrid --model models/agent_v2_chat_sft \
  --file-root . --memory-db agent.sqlite3 --seed 42 --temperature 0.8 --top-p 0.9
```

`file.read`はroot以下の通常UTF-8 fileだけを最大1 MiBまで読み、絶対path、path traversal、symlink脱出を拒否します。
`calculator.calculate`は四則演算、単項符号、括弧だけを解釈し、codeを実行しません。

長期memoryはSemantic、Episodic、Projectの3種です。同一type・本文はupsertされ、FTS5 trigramで日本語を検索します。
3 code point未満はparameter bindingされた完全一致・部分一致検索へ切り替わります。WAL、transaction、5秒busy timeout、
schema version 1 migrationを使用します。

`HybridReasoner`は通常会話だけを学習モデルへ渡し、解析・計画・評価・tool・memory・最終整形をruleへ委譲します。`ModelReasoner`は9 modeのID互換を維持し、実行時に使う7 modeを学習します。
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
  --output data/realpersona-chat-sft/train.jsonl --seed 42 --profile chat --split train
$M sft --train data/agent-v2-sft/train.jsonl --validation data/agent-v2-sft/validation.jsonl \
  --source models/agent-pretrain --output models/agent_v2_sft --steps 50000 --accumulation 4
$M resume --kind sft --train data/agent-v2-sft/train.jsonl \
  --validation data/agent-v2-sft/validation.jsonl --model models/agent_v2_sft --steps 1000
$M validate --kind sft --data data/agent-v2-sft/validation.jsonl \
  --model models/agent_v2_sft --batch-size 2
```

### 議論データを追加する場合（事前学習は再開不要）

`pretrain` / `pretrain-sharded` の実行中に議論機能を追加しても、事前学習を最初から
やり直す必要はありません。実行中の事前学習はそのまま完了させ、そのcheckpointを
`sft --source`へ渡します。議論の決定処理自体はC++で動くため、SFT未実行でも
`/compare JSON`は利用できます。

議論例だけを単独でSFTすると通常会話や他modeを忘れるおそれがあります。まず既存の
agent SFT corpusへ追加例を混ぜ、元bundleを上書きせず別の出力先へ学習します。

```sh
# 事前学習完了後。生成データ本文はdata/配下に置き、Gitへ追加しない
./scripts/generate_discussion_data.py data/discussion-reasoning \
  --seed 20260918 --groups-per-category 10

./scripts/mix_discussion_sft.py \
  --base data/agent-v3-base-sft/train.jsonl \
  --discussion data/discussion-reasoning/train.sft.jsonl \
  --output data/agent-v3-discussion-sft/train.jsonl --seed 20260918
./scripts/mix_discussion_sft.py \
  --base data/agent-v3-base-sft/validation.jsonl \
  --discussion data/discussion-reasoning/validation.sft.jsonl \
  --output data/agent-v3-discussion-sft/validation.jsonl --seed 20260918

# 最初は1 updateのsmoke。既存checkpointは変更しない
./build/model/agent_model_cli sft \
  --train data/agent-v3-discussion-sft/train.jsonl \
  --validation data/agent-v3-discussion-sft/validation.jsonl \
  --source models/agent_v3_pretrain_jawiki \
  --output models/agent_v3_discussion_smoke \
  --steps 1 --batch-size 1 --accumulation 1 --seed 20260918

# smokeのloss、勾配、checkpoint再ロードを確認後に別bundleへ本学習
./build/model/agent_model_cli sft \
  --train data/agent-v3-discussion-sft/train.jsonl \
  --validation data/agent-v3-discussion-sft/validation.jsonl \
  --source models/agent_v3_pretrain_jawiki \
  --output models/agent_v3_discussion_sft \
  --steps 50000 --batch-size 1 --accumulation 4 --seed 20260918
```

`data/agent-v3-base-sft`がまだない場合は、JMultiWOZのローカルpathと固定revisionを指定して先に生成します。

```sh
AI_CPP_SFT_PROFILE=agent \
AI_CPP_SFT_DATA_DIR=data/agent-v3-base-sft \
AI_CPP_JMULTIWOZ_DIR=/path/to/jmultiwoz \
AI_CPP_JMULTIWOZ_REVISION=FIXED_REVISION \
./scripts/run_04_generate_sft.sh
```

すでに学習済みのv3 SFT bundleがある場合は、それを
`--source`に指定して別bundleへ追加SFTして構いません。optimizer/RNGまで同じ
checkpointから厳密に継続する`resume`は指定directoryをその場で更新するため、保全用の
複製を作った場合だけ使用してください。

詳細な比較器の評価、再開条件、未実行項目は
[docs/discussion_evaluation.md](docs/discussion_evaluation.md) を参照してください。

SFTではmode tokenより前のprompt targetをPADにしてlossから除外します。agent profileはCHAT 20%、PARSE 15%、PLAN 15%、EVALUATE 10%、SUMMARIZE 20%、MEMORY_WRITE 10%、FINAL 10%でsampleします。MEMORY_QUERYとTOOLはID互換だけを維持します。manifestにはtrained_modes、sft_profile、生成seed、split fingerprintを記録し、checkpointにはweight/Adam fingerprint、step、shuffle RNG、
dropout counter、data/tokenizer fingerprintを保存します。`latest.json`はatomic renameで更新します。

内部会話要約の固定形式、ローカル変換、70/30混合、ライブドアablation、評価条件は [docs/conversation_summary.md](docs/conversation_summary.md) を参照してください。

`generate-sft`は`--profile chat|agent`と`--split train|validation`を必須の運用単位として扱います。agent profileは各構造化modeにつきtrain 2,000件、validation 200件をsplit固有のtemplateと語彙からローカル生成します。`validate --kind sft`はlossに加えて500件を自己回帰生成し、JSON 100%かつintent・tool名・arguments等の意味一致95%以上をmodel backend昇格条件として判定します。

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
`AI_CPP_TOKENIZER`、`AI_CPP_PRETRAIN_MODEL`、`AI_CPP_SFT_PROFILE`、`AI_CPP_SFT_MODEL`、
`AI_CPP_PRETRAIN_TOKENS`、`AI_CPP_VALIDATION_TOKENS`、`AI_CPP_SFT_STEPS` などの
環境変数で変更できます。既定の`AI_CPP_SFT_PROFILE=chat`は旧`agent_v1_sft`を上書きせず`agent_v2_chat_sft`へ保存し、`run_07_agent.sh`をhybridで起動します。第2段階は`AI_CPP_SFT_PROFILE=agent`で別の`agent_v2_sft`へ50,000 steps学習します。既定ではJawiki indexもtrain 15M tokens、validation 26万tokensまでに
制限し、resume時にも同じfingerprintになるよう同じ制限を適用します。
