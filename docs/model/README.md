# `ai/model` 公開ヘッダー設計書

`include/ai/model` の公開APIを、1ヘッダーにつき1文書で説明する。

| ヘッダー | 設計書 | 主な責務 |
|---|---|---|
| `agent_dataset.h` | [agent_dataset.md](agent_dataset.md) | 言語モデル用の学習窓とSFTコーパス |
| `agent_tokenizer.h` | [agent_tokenizer.md](agent_tokenizer.md) | SentencePiece tokenizerとmode token |
| `agent_training.h` | [agent_training.md](agent_training.md) | 学習、評価、checkpoint再開 |
| `agent_transformer.h` | [agent_transformer.md](agent_transformer.md) | causal Transformer本体 |
| `jawiki_sharding.h` | [jawiki_sharding.md](jawiki_sharding.md) | Wikipedia JSONLの再現可能な分割 |
| `model_language_model.h` | [model_language_model.md](model_language_model.md) | bundle保存・生成・Agent向けadapter |

依存の向きは概ね `dataset/tokenizer -> transformer/training -> bundle -> agent::ILanguageModel` である。NER bundleはこのモデルbundleとは独立している。
