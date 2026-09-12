# ai-cpp-from-scratch

C++20／CUDA で文字単位の会話 Transformer を学習・評価するプロジェクトです。行列演算、自動微分、Transformer、モデル保存、Adam をリポジトリ内で実装しています。前処理から学習・追加学習・評価・対話生成まで C++ の実行ファイルを使用し、Python は不要です。

## ディレクトリ構成

```text
.
├── model/                          学習・評価で共有するモデル関連コード
│   ├── include/                    Transformer・語彙・データ・保存 API
│   └── src/                        共通モデルの実装
├── train/
│   ├── main_train.cpp              学習・前処理の入口
│   ├── include/                    学習 API
│   └── src/                        新規学習・追加学習・CLI
├── validation/
│   ├── main_validation.cpp         評価・対話生成の入口
│   ├── include/                    評価・生成 API
│   └── src/                        評価・対話生成・CLI
├── lib/
│   ├── include/                    行列・自動微分・Optimizer などの API
│   └── src/                        基盤 C++／CUDA 実装
├── tests/                          自動テスト（fixtures/ は画像モデルなどの検証用コード）
├── docs/                           日本語設計仕様・検証記録
├── scripts/                        ビルド・テスト・実行スクリプト
├── third_party/                    同梱 JSON ライブラリ
├── data/                           ローカル配置のデータ
└── models/                         学習済みモデルの保存先
```

`model/` はソースコード、`models/` は重みなどの保存先です。`model/`・`train/`・`validation/` は同じ階層にあり、学習と評価の両方が共通モデルを使用します。

アプリの実行ファイルは `main_train` と `main_validation` の2つです。MNIST／CIFAR-10 の学習実行ファイルは廃止し、基盤の回帰テストが使用する画像モデルなどは `tests/fixtures/` に配置しています。

## 必要環境・ビルド

- CMake 3.18以降
- C++20対応コンパイラ
- NVIDIA GPU
- CUDA Toolkit（CUDA Runtime／cuBLAS／cuRAND）

CPU のみでの学習・評価には対応していません。JSON パーサー nlohmann/json 3.12.0（MIT）は `third_party/nlohmann/` に同梱しています。

リポジトリルートで実行します。

```sh
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j2
ctest --test-dir build/release --output-on-failure
```

既定の CUDA architecture は86です。別の GPU 向けには configure 時に `-DAI_CPP_CUDA_ARCHITECTURES=89` などを指定してください。

| 実行ファイル | 用途 |
| --- | --- |
| `build/release/main_train` | データの前処理、新規学習、保存モデルからの追加学習 |
| `build/release/main_validation` | 保存モデルの validation／test 評価、対話生成 |

## データの準備

RealPersonaChat の対話本文を使用します。

```sh
git clone https://github.com/nu-dialogue/real-persona-chat.git data/real-persona-chat
build/release/main_train prepare data/real-persona-chat data/conversation \
    "$(git -C data/real-persona-chat rev-parse HEAD)"
```

前処理で次のファイルを作成します。

```text
data/conversation/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── metadata.json
```

対話 ID 単位で seed42 の90／5／5%に分割します。出典・VERSION・revision・分割 ID は `metadata.json` に記録します。元データの利用条件は [公式 RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) を参照してください。

語彙は新規学習時に train 本文から構築し、未知文字は UNK に変換します。通常文字は Unicode コードポイント1個につき1 token。PAD／UNK／対話開始・終了／話者 A・B／発話終了に専用 ID を使い、両話者の本文と区切りを教師にします。ペルソナや属性は入力しません。既定の文脈長128では129 token の窓を128 tokenずつ進め、1 token先を教師にし、末尾を右 PAD で補います。

## 新規学習

```sh
build/release/main_train data/conversation models/conversation
```

主な学習オプションと既定値は次のとおりです。

| オプション | 既定値 | 意味 |
| --- | ---: | --- |
| `--epochs` | 10 | 学習 epoch 数（追加学習時は追加する回数） |
| `--batch` | 64 | バッチサイズ |
| `--lr` | 0.0003 | Adam 学習率 |
| `--clip` | 1.0 | 全体勾配 L2 norm の上限 |
| `--seed` | 42 | 新規学習のモデル初期化・dropout・shuffle 用 seed |
| `--max-batches` | 0 | 各 split／epoch のバッチ数上限。0は全件 |

モデルは FP32 の decoder-only Transformer です。学習可能な位置埋め込み、Pre-LayerNorm、因果 Attention、GELU、dropout を使用し、入出力の重みは共有しません。

| モデル構造オプション | 既定値 |
| --- | ---: |
| `--blocks` | 4 |
| `--embedding` | 256 |
| `--heads` | 4 |
| `--hidden` | 1024 |
| `--context` | 128 |
| `--dropout` | 0.1 |

明示的に設定する例です。

```sh
build/release/main_train data/conversation models/conversation \
    --epochs 10 --batch 64 --lr 0.0003 --clip 1 --seed 42 \
    --blocks 4 --embedding 256 --heads 4 --hidden 1024 --context 128 --dropout 0.1
```

短い動作確認では、バッチサイズと処理バッチ数を指定できます。

```sh
build/release/main_train data/conversation models/conversation-smoke \
    --epochs 1 --batch 32 --max-batches 3
```

各 epoch で train と validation の有効 token 平均 loss／perplexity を計算し、validation loss が改善した重みを保存します。最後に最良重みを読み直して test を評価します。`--max-batches` の正数は validation／test にも適用され、指標は部分データの結果になります。

## 保存モデルからの追加学習

元モデルとは別の、新規または空のディレクトリを出力先に指定します。

```sh
build/release/main_train data/conversation models/conversation-b32 \
    --from-model models/conversation \
    --batch 32 --epochs 10 --max-batches 0
```

重み・語彙・特殊 ID・モデル設定を引き継ぎます。語彙は再構築せず、追加データの未知文字は UNK にします。保存された構造を使用するため、`--blocks`、`--embedding`、`--heads`、`--hidden`、`--context`、`--dropout` の同時指定はエラーです。

batch、lr、clip、追加 epoch 数、max-batches は変更できます。追加学習の `--seed` は shuffle 用です。dropout は保存モデル設定の seed から新しく開始します。Adam は新規初期化するため、Optimizer・乱数状態を含む完全な中断再開ではありません。

更新前の validation を epoch 0 として評価し、読み込んだ重みを最初の最良候補として保存します。各追加 epoch で validation loss が改善した場合だけ置き換え、最後に最良重みで test を評価します。改善がなければ更新前の重みが残ります。

## 保存モデルの評価

```sh
# validation 全件の評価
build/release/main_validation data/conversation models/conversation \
    --split validation --batch 64 --max-batches 0

# test 全件の評価
build/release/main_validation data/conversation models/conversation \
    --split test --batch 64 --max-batches 0
```

既定は `--split validation --batch 64 --max-batches 0`。保存された設定・語彙で指定 split の有効 token 平均 loss／perplexity を計算し、バッチ数・有効 token 数などとともに JSON を標準出力へ返します。Optimizer 更新やモデル書き込みは行いません。

短い確認には `--batch 32 --max-batches 3` などを指定します。単独評価では対象 split の JSONL と保存モデルが必要です。

## 対話生成

```sh
build/release/main_validation chat models/conversation \
    --temperature 0.8 --top-k 40 --max-tokens 128 --seed 42
```

上記は既定値です。利用者は `A>` に入力し、モデルは `B>` として返答します。直近 `min(128, 保存モデルの文脈長)` token を再計算し、発話終了・対話終了または指定 token 数で停止します。`--max-tokens` は1～128。終了は EOF（Ctrl-D）です。KV cache は使用しません。生成時は元の学習データを必要としません。

## 保存ファイル・ログ

```text
models/conversation/
├── weights.bin                     最良モデルの重み
├── manifest.json                   bundle v1 の設定・語彙・特殊 ID
└── metrics.jsonl                   学習条件と各 split／epoch の指標
```

学習条件、読み込み元、loss／perplexity、処理時間、tokens/s、GPU 使用メモリなどを標準出力と `metrics.jsonl` に記録します。GPU メモリ値は各バッチ終了時のデバイス全体のサンプル値で、演算途中の厳密な最大値ではありません。

`weights.bin` と `manifest.json` を一緒に移動すれば、評価・対話生成・再追加学習に使用できます。Optimizer、dropout の乱数状態、shuffle の途中状態は保存しません。

## テスト・実行スクリプト

`scripts/run_train.sh` と `scripts/run_validation.sh` は、先頭の設定変数を編集して実行できます。同名の環境変数でも上書きできます。相対パスはリポジトリルート基準です。既定では Release の対象実行ファイルをビルドしてから実行します。

```sh
# 設定変数に従って学習・評価
./scripts/run_train.sh
./scripts/run_validation.sh

# 少数バッチで新規学習
EPOCHS=1 BATCH_SIZE=32 MAX_BATCHES=3 OUTPUT_DIR=models/conversation-smoke \
    ./scripts/run_train.sh

# 追加学習（出力先は元モデルとは別の新規または空ディレクトリ）
FROM_MODEL=models/conversation OUTPUT_DIR=models/conversation-b32 BATCH_SIZE=32 \
    ./scripts/run_train.sh

# test 評価・対話生成
SPLIT=test BATCH_SIZE=32 ./scripts/run_validation.sh
MODE=chat MAX_TOKENS=64 ./scripts/run_validation.sh

# ビルド・学習を行わず、実行コマンドだけ確認
DRY_RUN=1 ./scripts/run_train.sh
```

学習用では `DATA_DIR`、`OUTPUT_DIR`、`FROM_MODEL`、`EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`、`CLIP_NORM`、`SEED`、`MAX_BATCHES` を変更できます。構造設定の `BLOCKS`、`EMBEDDING`、`HEADS`、`HIDDEN`、`CONTEXT`、`DROPOUT` は新規学習専用です。追加学習では保存設定を使用します。

評価用は `MODE=evaluate`（既定）で `DATA_DIR`、`MODEL_DIR`、`SPLIT`、`BATCH_SIZE`、`MAX_BATCHES` を使用します。`MODE=chat` では `MODEL_DIR`、`TEMPERATURE`、`TOP_K`、`MAX_TOKENS`、`SEED` を使用します。

両スクリプト共通で `BUILD_DIR`、`BUILD_TYPE`、`BUILD_JOBS` を変更でき、`RUN_BUILD=0` でビルドを省略、`DRY_RUN=1` でコマンド表示のみにできます。全テストは次の `run.sh` または CTest で実行します。


CTest は行列・自動微分・モデル保存などの基盤と、会話データ・学習・追加学習・評価・生成を検証します。検証範囲と実データの GPU 実測は [検証記録](docs/conversation_validation.md) を参照してください。

```sh
# Debug ビルドと全テスト
./scripts/run.sh

# Debug ビルド・全テストの後に指定処理を実行
./scripts/run.sh train data/conversation models/conversation-smoke \
    --epochs 1 --batch 32 --max-batches 3
./scripts/run.sh validation data/conversation models/conversation --split test
./scripts/run.sh validation chat models/conversation
```

API の形状・所有権・例外・保存互換性は [設計仕様一覧](docs/README.md) と [会話 API 設計仕様](docs/conversation_runtime.md) を参照してください。

## GPU メモリの再利用

既定で CUDA メモリプールを使用し、使い終わった領域を実行中は保持して再利用します。
`MEMORY_POOL=0 bash scripts/run_train.sh` / `MEMORY_POOL=0 bash scripts/run_validation.sh` で従来方式に切り替えられます。
直接起動する場合は `AI_CPP_CUDA_MEMORY_POOL=0` を指定します（既定1、初回確保時に固定）。非対応 GPU / ドライバーは理由を stderr に表示して従来方式へ切り替えます。

予約済みメモリは計算に使用中とは限りません。学習の `pool_used_bytes` / `pool_reserved_bytes` と従来のデバイス全体の指標を区別してください。
設定は次回起動から適用されます。詳細と検証・性能測定手順は [CUDA メモリ管理](docs/cuda_memory.md) を参照してください。
