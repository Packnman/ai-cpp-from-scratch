# `ai/ner` 公開ヘッダー設計書

`include/ai/ner` の公開APIを、1ヘッダーにつき1文書で説明する。

| ヘッダー | 設計書 | 主な責務 |
|---|---|---|
| `entity.h` | [entity.md](entity.md) | entity型、UTF-8 byte span、code point変換 |
| `extractor.h` | [extractor.md](extractor.md) | rule/model/hybrid抽出interface |

NERは候補抽出器であり、発言の真偽、否定、提案、訂正、現在値の確定は `ai/agent` の状態更新責務である。
