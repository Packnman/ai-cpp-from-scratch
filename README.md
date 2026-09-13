# ai-cpp-from-scratch

C++20／CUDA で文字単位またはサブワードの会話 Transformer を学習・評価するプロジェクトです。行列演算、自動微分、Transformer、モデル保存、Adam をリポジトリ内で実装しています。前処理から学習・追加学習・評価・対話生成まで C++ の実行ファイルを使用し、Python は不要です。

## 開発目標とロードマップ

自作 C++／CUDA モデルを、日常会話、物理の説明、利用者の資料に基づく対話の順で育てます。目標とする回答形式は Markdown＋LaTeX です。チェック済みの項目は実装・検証済み、未チェックの項目は今後の計画です。後続の操作手順と既定値は現在の実装を示します。

### ① 学習基盤の整備

実装・検証済み。RTX 3060 Ti での測定結果と再実行手順は [学習基盤の検証記録](docs/training_foundation.md) を参照してください。

**現在できていること**

- [x] 新規学習、保存モデルへの追加学習、評価、対話生成。
- [x] GPU メモリプールによる再利用。
- [x] 重み・語彙・モデル設定の保存。`--from-model` の追加学習では Adam を新規初期化します。
- [x] epoch checkpoint に重み、Adam、shuffle、dropout 状態を保存し、`--resume` で完全再開します。

**実施項目**

- [x] C++ から利用する SentencePiece の BPE を第一候補として、サブワード tokenizer を導入する。
- [x] 日本語・英数字・LaTeX・Markdown を扱い、未知文字のバイト分解と、改行・空白・数式記号を保持する設定を検証する。
- [x] tokenizer は学習用データだけから構築する。評価データと、最後に参照させる利用者の資料は使用しない。
- [x] tokenizer を変更するモデルは新規学習する。その後の追加学習では tokenizer を固定する。
- [x] tokenizer 本体をモデルと一緒に保存する形式を追加し、既存の文字単位モデルも読み込めるようにする。
- [x] 新規モデルの文脈長512を測定の出発点とする。生成側に残る履歴128 token の固定制限も見直し、入力文脈長と出力長の設定を分ける。
- [x] RTX 3060 Ti で速度とメモリを測定する。バッチ32から試し、メモリ不足なら16、8、4、2、1と下げる。測定は他の GPU 学習がない状態で行う。

**完了条件**

- [x] 日本語と Markdown＋LaTeX の往復変換、保存・読込・追加学習、512 token の入力での学習・生成が動作する。
- [x] 3バッチのウォームアップ後に10バッチを各3回測定し、設定、tokens/s、使用中・予約済みメモリを記録する。

### ② 日常会話の学習

- [ ] RealPersonaChat を出発点に、サブワードモデルを新規学習する。
- [ ] 学習に使わない固定の会話評価セットを用意し、各モデルの返答を保存する。
- [ ] 質問への適合、数往復の文脈維持、反復、応答終了を確認する。loss だけで完成と判断しない。
- [ ] 不十分な場合はデータの質と量を確認し、日本語文章による基礎学習やモデル規模の拡大を検討する。

**次段階への条件**

- [ ] 固定評価セットで上記の観点を採点し、日常会話の基準モデルとして使えるかを実際の返答から判断する。学習回数だけを移行条件にしない。

### ③ 物理の追加学習

- [ ] 初期範囲は高校力学とする。
- [ ] 「質問・説明・途中式・答え」を持つ、内容を確認した教材を用意する。
- [ ] 本文は Markdown、文中数式は `$...$`、独立数式は `$$...$$` に統一する。
- [ ] 日常会話データを混ぜて追加学習し、会話能力の低下を継続して確認する。
- [ ] 評価問題は元問題単位で学習データから分離し、数値だけを変えた問題の混入も避ける。

**完了条件**

- [ ] 未学習問題について、物理法則の選択、途中式、数値、単位、適用条件を評価する。
- [ ] 数式の表示形式と内容の正確さは別々に確認し、日常会話モデルとの比較結果も残す。

### ④ 利用者の資料に基づく対話

- [ ] 数枚の資料を、ページ番号付きの文章と LaTeX に変換する。図の条件・関係も文章化し、数式の変換結果は原本と照合する。
- [ ] 「回答指示・関連資料・会話履歴・質問」を入力する。
- [ ] 資料全文が文脈長に収まる場合は全文を使用する。長い場合は関連箇所を検索して渡す。
- [ ] 別の練習資料で、資料に基づく回答、参照ページの提示、情報不足の表明を学習・評価する。
- [ ] 利用者の資料自体を暗記させる追加学習は、初期実装では行わない。

**完了条件**

- [ ] 資料から答えられる質問、記載のない質問、前の会話を踏まえる質問を検証する。
- [ ] 回答内容と参照ページが対応し、Markdown＋LaTeX で出力できることを確認する。

### 共通の判断基準と前提

- 小さな自作モデルで最終目標を達成できる保証はありません。各段階の評価を残し、問題があれば次段階より改善を優先します。
- 文脈長512は最初の測定条件であり、資料対話に十分な長さとは決めつけません。
- データ追加、モデル拡大、文脈長拡張は、失敗例と速度・メモリ測定を根拠に判断します。
- モデルは Markdown＋LaTeX の文字列を生成します。画面上の数式描画は表示側の別機能として扱います。
- 実行中の学習は停止・変更せず、新しい構成は別の出力先で開始します。

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
- Git（初回 configure 時の SentencePiece 取得用）
- NVIDIA GPU
- CUDA Toolkit（CUDA Runtime／cuBLAS／cuRAND）

SentencePiece v0.2.1（Apache-2.0）は固定コミットから CMake が取得して静的リンクします。初回 configure にはネットワーク接続が必要です。取得済みソースを使う場合は `-DFETCHCONTENT_SOURCE_DIR_SENTENCEPIECE=/absolute/path/to/sentencepiece` を指定できます。

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

既定の文字単位 tokenizer の語彙は新規学習時に train 本文から構築し、未知文字は UNK に変換します。通常文字は Unicode コードポイント1個につき1 token。PAD／UNK／対話開始・終了／話者 A・B／発話終了に専用 ID を使い、両話者の本文と区切りを教師にします。ペルソナや属性は入力しません。既定の文脈長128では129 token の窓を128 tokenずつ進め、1 token先を教師にし、末尾を右 PAD で補います。

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

### サブワード・文脈長512の新規モデル

①の測定構成は次のコマンドで指定します。CLI の既定値（文字単位、文脈長128）は維持しています。tokenizer を切り替えるときは新規学習が必要です。新規学習も出力先を新規または空のディレクトリに限定します。

```sh
build/release/main_train data/conversation models/conversation-bpe-512 \
    --tokenizer bpe --vocab-size 4096 --context 512 \
    --batch 16 --epochs 10
```

SentencePiece BPE は train の発話本文だけから構築します。validation／test と利用者の資料は使用しません。未知文字は UTF-8 バイトへ分解し、空白・改行・タブ・Markdown・LaTeX の文字列を保持します。語彙数4096は特殊 ID と256個のバイト token を含む目標値で、小規模データでは実際の語彙数が小さくなることがあります。

バッチ32からの自動縮小と3回の測定は `bash scripts/run_context_benchmark.sh BUNDLE data/conversation/train.jsonl` で実行できます。測定手順・設定・結果は [学習基盤の検証記録](docs/training_foundation.md) を参照してください。

## 保存モデルからの追加学習

元モデルとは別の、新規または空のディレクトリを出力先に指定します。

```sh
build/release/main_train data/conversation models/conversation-b32 \
    --from-model models/conversation \
    --batch 32 --epochs 10 --max-batches 0
```

重み・語彙・特殊 ID・モデル設定を引き継ぎます。tokenizer は再構築しません。追加データの未知文字は文字単位モデルでは UNK、BPE モデルではバイト token にします。保存された構造を使用するため、`--blocks`、`--embedding`、`--heads`、`--hidden`、`--context`、`--dropout`、`--tokenizer`、`--vocab-size` の同時指定はエラーです。

batch、lr、clip、追加 epoch 数、max-batches は変更できます。追加学習の `--seed` は shuffle 用です。dropout は保存モデル設定の seed から新しく開始します。Adam は新規初期化するため、Optimizer・乱数状態を含む完全な中断再開ではありません。

更新前の validation を epoch 0 として評価し、読み込んだ重みを最初の最良候補として保存します。各追加 epoch で validation loss が改善した場合だけ置き換え、最後に最良重みで test を評価します。改善がなければ更新前の重みが残ります。

## checkpoint からの完全再開

同じモデルディレクトリの最新 checkpoint から、追加する epoch 数を指定して再開します。

```sh
build/release/main_train data/conversation models/conversation --resume --epochs 10
build/release/main_train data/conversation models/conversation --resume --epochs 10 --lr 0.0001
```

checkpoint が epoch 30 なら、上記は epoch 31～40 を実行します。`--lr` を省略すると保存値を使い、指定時は Adam の moment と step を保ったまま学習率だけを変更します。`--resume` では `--batch`、`--clip`、`--seed`、`--max-batches`、モデル構造、tokenizer を変更できず、`--from-model` とも併用できません。データ、tokenizer、モデル構造、checkpoint ファイルの不一致や破損は更新前に拒否します。

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

上記は既定値です。利用者は `A>` に入力し、モデルは `B>` として返答します。直近の入力文脈を再計算し、発話終了・対話終了または指定 token 数で停止します。`--input-context` は0（既定：保存モデルの文脈長）または保存モデルの文脈長以下の正数、`--max-tokens` は出力長の正数です。履歴と出力の固定128 token 制限はありません。終了は EOF（Ctrl-D）です。KV cache は使用しません。生成時は元の学習データを必要としません。

## 保存ファイル・ログ

```text
models/conversation/
├── weights.bin                     最良モデルの重み
├── manifest.json                   設定・特殊 ID（v1: 文字語彙、v2: tokenizer 情報）
├── tokenizer.model                 BPE の場合のみ：固定 tokenizer 本体
├── checkpoint/
│   ├── latest.json                 最新の完了 checkpoint を指す commit ファイル
│   ├── epoch-N.weights.bin          最新 epoch の重み
│   ├── epoch-N.adam.bin             Adam moment・step・学習率
│   └── epoch-N.json                 epoch・乱数・条件・fingerprint
└── metrics.jsonl                   学習条件と各 split／epoch の指標
```

学習条件、読み込み元、loss／perplexity、処理時間、tokens/s、GPU 使用メモリなどを標準出力と `metrics.jsonl` に記録します。GPU メモリ値は各バッチ終了時のデバイス全体のサンプル値で、演算途中の厳密な最大値ではありません。

`weights.bin` は従来どおり validation 最良モデルで、評価・対話生成・`--from-model` に使用します。`checkpoint/` は最新 epoch の重み、Adam、shuffle、各 dropout seed カウンタを保持し、完全再開に使用します。checkpoint は `latest.json` を最後に rename して確定し、確定後は直前世代を削除するため最新1世代だけを保持します。`metrics.jsonl` は再開時に追記されます。

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

# checkpoint からの完全再開は main_train を直接実行
build/release/main_train data/conversation models/conversation \
    --resume --epochs 10

# test 評価・対話生成
SPLIT=test BATCH_SIZE=32 ./scripts/run_validation.sh
MODE=chat MAX_TOKENS=64 ./scripts/run_validation.sh

# ビルド・学習を行わず、実行コマンドだけ確認
DRY_RUN=1 ./scripts/run_train.sh
```

学習用スクリプトでは `DATA_DIR`、`OUTPUT_DIR`、`FROM_MODEL`、`EPOCHS`、`BATCH_SIZE`、`LEARNING_RATE`、`CLIP_NORM`、`SEED`、`MAX_BATCHES` を変更できます。構造設定の `BLOCKS`、`EMBEDDING`、`HEADS`、`HIDDEN`、`CONTEXT`、`DROPOUT`、`TOKENIZER`、`VOCAB_SIZE` は新規学習専用です。追加学習では保存設定を使用します。`scripts/run_train.sh` は新規学習と `--from-model` に対応し、checkpoint の `--resume` は上記のように `main_train` を直接実行します。

評価用は `MODE=evaluate`（既定）で `DATA_DIR`、`MODEL_DIR`、`SPLIT`、`BATCH_SIZE`、`MAX_BATCHES` を使用します。`MODE=chat` では `MODEL_DIR`、`TEMPERATURE`、`TOP_K`、`MAX_TOKENS`、`INPUT_CONTEXT`、`SEED` を使用します。

両スクリプト共通で `BUILD_DIR`、`BUILD_TYPE`、`BUILD_JOBS` を変更でき、`RUN_BUILD=0` でビルドを省略、`DRY_RUN=1` でコマンド表示のみにできます。全テストは次の `run.sh` または CTest で実行します。


CTest は行列・自動微分・モデル保存などの基盤と、会話データ・学習・追加学習・checkpoint 再開・評価・生成を検証します。サブワード、文脈長512、GPU 実測については [学習基盤の検証記録](docs/training_foundation.md) を参照してください。

```sh
# Debug ビルドと全テスト
./scripts/run.sh

# Debug ビルド・全テストの後に指定処理を実行
./scripts/run.sh train data/conversation models/conversation-smoke \
    --epochs 1 --batch 32 --max-batches 3
./scripts/run.sh validation data/conversation models/conversation --split test
./scripts/run.sh validation chat models/conversation
```

現在の文書一覧と主要 API の参照先は [docs/README.md](docs/README.md) を参照してください。

## GPU メモリの再利用

既定で CUDA メモリプールを使用し、使い終わった領域を実行中は保持して再利用します。
`MEMORY_POOL=0 bash scripts/run_train.sh` / `MEMORY_POOL=0 bash scripts/run_validation.sh` で従来方式に切り替えられます。
直接起動する場合は `AI_CPP_CUDA_MEMORY_POOL=0` を指定します（既定1、初回確保時に固定）。非対応 GPU / ドライバーは理由を stderr に表示して従来方式へ切り替えます。

予約済みメモリは計算に使用中とは限りません。学習の `pool_used_bytes` / `pool_reserved_bytes` と従来のデバイス全体の指標を区別してください。
設定は次回起動から適用されます。検証・性能測定手順と実測値は [学習基盤の検証記録](docs/training_foundation.md) を参照してください。
