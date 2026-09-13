# 文書一覧

現在の利用手順と実装状況は、リポジトリルートの [README](../README.md) を正とする。このディレクトリには、再現に必要な検証条件と補足情報だけを置く。

| 文書 | 内容 |
| --- | --- |
| [training_foundation.md](training_foundation.md) | SentencePiece BPE、文脈長512、GPU性能・メモリ測定、checkpoint完全再開の検証記録 |

## 主要な実装参照先

| 領域 | 公開API | 実装・入口 |
| --- | --- | --- |
| 学習・追加学習・再開 | `train/include/train.h` | `train/src/train.cpp`、`train/main_train.cpp` |
| 評価・対話生成 | `validation/include/validation.h` | `validation/src/validation.cpp`、`validation/main_validation.cpp` |
| Transformer | `model/include/model_transformer.h` | `model/src/model_transformer.cpp` |
| tokenizer・会話bundle | `model/include/tokenizer_conversation.h` | `model/src/tokenizer_conversation.cpp`、`model/src/tokenizer_subword.cpp` |
| Adam・状態保存 | `lib/include/optimizer_adam.h` | `lib/src/optimizer_adam.cpp` |
| CUDAメモリ | `lib/include/cuda_memory.h` | `lib/src/cuda_memory.cpp` |

## 学習方法の区別

| 方法 | 出力先 | 引き継ぐ状態 |
| --- | --- | --- |
| 新規学習 | 新規または空のディレクトリ | なし |
| `--from-model SOURCE` | SOURCEとは別の新規または空のディレクトリ | 重み、tokenizer、モデル設定。Adamと乱数状態は新規 |
| `--resume` | checkpointを持つ同じモデルディレクトリ | 最新epochの重み、Adam、shuffle、dropout、最良モデル情報 |

checkpointは各epochの完了後に作られる。`weights.bin` はvalidation lossが最良だった評価・生成用モデル、`checkpoint/epoch-N.weights.bin` は完全再開用の最新epochモデルであり、用途が異なる。

## checkpointの検証範囲

`tests/conversation_check.cpp` は次を検証する。

- 連続2 epochと、1 epoch後にcheckpointから再開した1 epochの状態が一致する。
- モデル重み、Adam、shuffle、dropout、epoch番号、最良validation情報を復元する。
- `metrics.jsonl` を置き換えず、`resume_start` を追記する。
- `--lr` の変更時もAdamのmomentとstepを維持する。
- checkpointの破損や学習データの変更を、更新開始前に拒否する。

## コーディングスタイル

- インデントは半角スペース4個とし、タブ文字は使用しない。
- `.editorconfig` と `.clang-format` をエディタ・formatterの基準にする。
