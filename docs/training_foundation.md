# 学習基盤（①）の実装・検証記録

2026-09-12 UTC。SentencePiece BPE、tokenizer 保存互換性、512 token の学習・生成と速度・メモリを検証した。これは基盤の動作確認であり、会話品質の評価は②で行う。

## 実装と検証範囲

- SentencePiece v0.2.1、固定 commit `31646a467d2051eb904e0b45de3a73e91fe1c1e3`。C++ API から train の発話だけで構築。Python はアプリ・学習に不要。
- BPE / byte fallback / identity 正規化。空白・改行・タブと文字としての `▁` は byte token で保持する。実装は `model/include/tokenizer_conversation.h` と `model/src/tokenizer_subword.cpp` を参照。
- bundle v1（文字単位）を維持し、v2（BPE）に `tokenizer.model` を追加。追加学習で tokenizer のバイナリが不変であることを確認。
- CLI は `--tokenizer bpe --vocab-size 4096 --context 512`。既定の文字単位と文脈長128は変更していない。入力窓 `--input-context` と出力長 `--max-tokens` を分離し、固定128 token 上限を撤去。
- 新規学習・追加学習とも出力先を新規または空に限定。既存モデルと既存ビルドは上書きしていない。
- 全18件の CTest が通過。`subword_check` は日本語、Markdown＋LaTeX、未知 Unicode、空白・制御文字、特殊 ID 非注入、保存復元、評価データ非混入、追加学習、512 token 入力と129 token 出力を確認する。既存 `conversation_check` は文字単位 bundle 互換性を検証する。

## checkpoint と完全再開

2026-09-13 UTC 時点で、各 epoch の完了後に `checkpoint/` へ次の状態を保存する。

- epoch 終了時のモデル重みと Adam の moment・step・学習率。
- shuffle 用 `std::mt19937` の状態と、各 dropout の seed counter。
- 完了 epoch、最良 validation loss と epoch、学習条件、モデル構造。
- train／validation／test、tokenizer、重み、Adam ファイルの fingerprint。

`checkpoint/latest.json` は metadata、重み、Adam を書き終えた後に rename して確定する。確定後は直前の checkpoint を削除し、最新の完了 epoch だけを保持する。通常の `weights.bin` は validation loss が最良だった評価・生成用モデルであり、checkpoint の最新 epoch 重みとは役割が異なる。

再開は `main_train DATA MODEL --resume --epochs N` で行う。`--lr` を省略すると保存値を使い、指定した場合も Adam の moment と step は維持する。再開時は保存済みのバッチサイズ、勾配 clip、seed、`max-batches`、モデル構造、tokenizer を使い、変更は拒否する。データや保存ファイルの fingerprint が一致しない場合も、学習を更新する前に停止する。

`conversation_check` では、2 epoch の連続学習と「1 epoch＋再開1 epoch」の checkpoint 重み・Adam・乱数状態が一致すること、epoch 番号と metrics の追記、学習率だけの変更、checkpoint 破損、学習データ変更の拒否を検証する。

## 実測条件

- NVIDIA GeForce RTX 3060 Ti / 8 GiB、Driver 580.102.01（nvidia-smi 表示）、CUDA Toolkit 12.8.93、GNU C++ 13.3.0、Release、CUDA architecture 86。
- 実測開始前に nvidia-smi の compute process 一覧と OS の学習プロセス一覧を確認。各測定前にもスクリプトが compute process の不在を確認した。Xwayland の画面表示プロセスは稼働。
- FP32、4 blocks、embedding 256、4 heads、hidden 1024、context 512、dropout 0.1、語彙4096、モデル seed 42。
- Adam を新規初期化（lr 0.0003、clip 1）、shuffle seed 42。train split 14,482窓を固定 shuffle。右 PAD は学習計算に含むが tokens/s の分子から除外する。
- 既存モデルとは別の `models/conversation_stage1_bpe_512` を新規作成。train tokenizer 構築後に1バッチだけ学習した動作確認用モデルであり、会話用の完成モデルではない。測定は毎回同じ保存重みから新しいプロセスで始め、測定中の更新重みは保存しない。
- バッチ32を試すと `CUDA memory allocate: out of memory`。新しいプロセスで16へ下げたところ成功。8以下への縮小は不要だった。
- 各回3バッチのウォームアップ後、10バッチを測定。CUDA 同期後に計時を開始し、終了時にも同期している。モデル読込・tokenize・ウォームアップ時間は測定秒数に含まない。

## 測定結果

| 回 | 秒 | 有効 token | tokens/s | 使用中 bytes | 予約済み bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.682133 | 57025 | 15486.95 | 86269952 | 5402263552 |
| 2 | 3.689627 | 57025 | 15455.49 | 86269952 | 5402263552 |
| 3 | 3.664789 | 57025 | 15560.24 | 86269952 | 5402263552 |

平均 **15500.89 tokens/s**。予約済みメモリは **5.03125 GiB**。使用中メモリは **82.27 MiB** で、計算グラフとバッチの一時領域を解放し、同期した測定終了時の値。演算途中の使用中メモリの最大値ではない。予約済みには再利用待ちの領域を含む。両方ともこのプロセスの CUDA pool の値であり、画面表示などの使用量を含む GPU 全体の値とは異なる。

入力長512が資料対話に十分とは判断していない。今回のバッチ16はこのモデル・語彙数・GPUでの測定結果であり、構成を拡大した場合は再測定する。

## 再実行

初回 configure は SentencePiece のダウンロードにネットワークを使う。取得済みの v0.2.1 ソースがある場合は `-DFETCHCONTENT_SOURCE_DIR_SENTENCEPIECE=/absolute/path/to/sentencepiece` を追加できる。

```sh
cmake -S . -B build/stage1 -DCMAKE_BUILD_TYPE=Release
cmake --build build/stage1 -j2
ctest --test-dir build/stage1 --output-on-failure

# 新規または空の出力先を使う。max-batches は検証・test にも適用。
build/stage1/main_train data/conversation models/stage1-smoke \
    --tokenizer bpe --vocab-size 4096 --context 512 \
    --batch 1 --epochs 1 --max-batches 1

# 他の GPU 学習がない状態で実行。stdout は JSONL、stderr は試行/OOM記録。
BUILD_DIR=build/stage1 bash scripts/run_context_benchmark.sh \
    models/stage1-smoke data/conversation/train.jsonl \
    > stage1-benchmark.jsonl 2> stage1-benchmark.log

# 入力文脈長と出力長は独立に指定する。
build/stage1/main_validation chat models/stage1-smoke \
    --input-context 512 --max-tokens 256
```

`run_context_benchmark.sh` は32→16→8→4→2→1の順に縮小し、成功したバッチで3回測定する。CUDA の out of memory 以外のエラーでは停止する。各回は13フルバッチ以上の train 窓が必要。GPU を使用する別プロセスが検出された場合は停止し、そのプロセスを終了させない。

このセッションのビルドは `/tmp/ai-cpp-stage1-build`。生ログは `/tmp/ai-cpp-stage1-benchmark.jsonl` と `/tmp/ai-cpp-stage1-benchmark.log`。次の JSON とハッシュは再確認用に文書へ保存する。

```json
{"allocator": "pool", "batch_size": 16, "blocks": 4, "bundle": "models/conversation_stage1_bpe_512", "clip_norm": 1.0, "context": 512, "dropout": 0.10000000149011612, "dtype": "float32", "embedding": 256, "heads": 4, "hidden": 1024, "learning_rate": 0.0003, "measured_batches": 10, "model_seed": 42, "pool_reserved_bytes": 5402263552, "pool_used_bytes": 86269952, "seconds": 3.682133186, "shuffle_seed": 42, "tokenizer": "bpe", "tokens_per_second": 15486.946593028533, "train_jsonl": "data/conversation/train.jsonl", "valid_tokens": 57025, "vocabulary": 4096, "warmup_batches": 3, "warmup_reserved_bytes": 5402263552}
{"allocator": "pool", "batch_size": 16, "blocks": 4, "bundle": "models/conversation_stage1_bpe_512", "clip_norm": 1.0, "context": 512, "dropout": 0.10000000149011612, "dtype": "float32", "embedding": 256, "heads": 4, "hidden": 1024, "learning_rate": 0.0003, "measured_batches": 10, "model_seed": 42, "pool_reserved_bytes": 5402263552, "pool_used_bytes": 86269952, "seconds": 3.689626608, "shuffle_seed": 42, "tokenizer": "bpe", "tokens_per_second": 15455.493484450717, "train_jsonl": "data/conversation/train.jsonl", "valid_tokens": 57025, "vocabulary": 4096, "warmup_batches": 3, "warmup_reserved_bytes": 5402263552}
{"allocator": "pool", "batch_size": 16, "blocks": 4, "bundle": "models/conversation_stage1_bpe_512", "clip_norm": 1.0, "context": 512, "dropout": 0.10000000149011612, "dtype": "float32", "embedding": 256, "heads": 4, "hidden": 1024, "learning_rate": 0.0003, "measured_batches": 10, "model_seed": 42, "pool_reserved_bytes": 5402263552, "pool_used_bytes": 86269952, "seconds": 3.664788765, "shuffle_seed": 42, "tokenizer": "bpe", "tokens_per_second": 15560.241983005533, "train_jsonl": "data/conversation/train.jsonl", "valid_tokens": 57025, "vocabulary": 4096, "warmup_batches": 3, "warmup_reserved_bytes": 5402263552}
```

### 入力・モデル SHA-256

- `data/conversation/train.jsonl`: `ad70b6eec2b9d696c29ad125feef322dc060bc33956225824500e60a743270dd`
- `models/conversation_stage1_bpe_512/weights.bin`: `34fb5ac442d72962b486e3652510a3234bc26f0eaa9dc7691702277a9327bc13`
- `models/conversation_stage1_bpe_512/tokenizer.model`: `c5a2d05f493f6d93a7a7fa0beed6c2073f72fad3f0b8d1418dd848386b987bb9`
