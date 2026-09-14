# ai_cpp Agent MVP

小さな推論器、短いcontext、外部memory、反復実行を組み合わせるC++20製AgentのMVPです。設計の正本は
[`docs/codex_agent_model_context.md`](docs/codex_agent_model_context.md)です。

現在の実装はCPUだけで動く決定論的なrule backendです。日本語生成品質を担う学習済みモデルではありません。
Transformer backend、multi-task SFT、Vision、実Robot制御、外部通信toolは次段階です。

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

sanitizer buildは`-DAI_CPP_ENABLE_SANITIZERS=ON`、CUDA環境では`-DAI_CPP_BUILD_CUDA_LIB=ON`を追加します。

## CLI

`./build/agent_cli [FILE_ROOT] [MEMORY_DB]`で起動します。省略時のfile rootは現在directory、
SQLite保存先は`./agent_memory.sqlite3`です。

```text
/calc (2 + 3) * 4
/read README.md
/remember project AgentではC++20を使う
/recall C++20
/quit
```

`file.read`はroot以下の通常UTF-8 fileだけを最大1 MiBまで読み、絶対path、path traversal、symlink脱出を拒否します。
`calculator.calculate`は四則演算、単項符号、括弧だけを解釈し、codeを実行しません。

長期memoryはSemantic、Episodic、Projectの3種です。同一type・本文はupsertされ、FTS5 trigramで日本語を検索します。
3 code point未満はparameter bindingされた完全一致・部分一致検索へ切り替わります。WAL、transaction、5秒busy timeout、
schema version 1 migrationを使用します。

`ModelReasoner`は注入した`ILanguageModel`を使い、設計正本にある9個のspecial modeを切り替えます。
構造化出力は必須fieldを検査し、不正時は修正を一度だけ要求します。将来は小型decoder-only Transformerを接続し、
各modeのmulti-task datasetでSFTします。旧model bundleとの互換性はありません。
