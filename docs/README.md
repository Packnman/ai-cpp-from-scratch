# ai_cpp ドキュメント

このディレクトリは、現在の実装に対応するアーキテクチャ、公開ヘッダー、学習、
評価手順の索引である。記述と実装が食い違う場合は、現在のヘッダー、実装、テストを
優先し、コード変更と同時に該当文書を更新する。

## 最初に読む文書

- [Agent全体構想](agent_overview.md): 小型生成モデル、決定的処理、NER、記憶、
  要約、限定Reasoningを組み合わせた現在の構成
- [Agent UML](agent_uml.md): 主要クラスの関係と処理sequence
- [Project Context](codex_agent_model_context.md): 初期目標、設計原則、背景
- [Project Structure](project_structure.md): ソース、テスト、学習、生成物の配置規則

## 現在のフォルダ構成

```text
docs/
├── README.md                    この索引
├── agent/                       include/ai/agent のヘッダー別設計書
├── model/                       include/ai/model のヘッダー別設計書
├── ner/                         include/ai/ner のヘッダー別設計書
├── lib/                         lib/include の数値・CUDA基盤設計書
├── train/                       汎用trainerの設計書
├── validation/                  実測を含む検証記録
├── agent_overview.md            現行Agentの全体設計
├── agent_uml.md                 クラス図・sequence図
├── conversation_summary.md      内部要約の形式、学習、評価
├── discussion_reasoning.md      根拠付き比較・判断の設計と実装計画
├── discussion_evaluation.md     限定Reasoningの実行・学習・評価
├── japanese_ner.md              日本語NERのデータ、学習、統合、評価
└── codex_agent_model_context.md 初期構想と設計背景
```

## 公開ヘッダー別設計書

### Agent

[agent/README.md](agent/README.md)を入口として、次の8ヘッダーを説明する。

- [`agent.h`](agent/agent.md): 1ターンのオーケストレーション
- [`components.h`](agent/components.md): component interfaceと既定実装
- [`context_builder.h`](agent/context_builder.md): prompt構築とtoken予算
- [`discussion.h`](agent/discussion.md): 根拠付き比較、制約、訂正状態
- [`memory.h`](agent/memory.md): SQLite長期記憶
- [`reasoner.h`](agent/reasoner.md): rule/model/hybrid backend
- [`tools.h`](agent/tools.md): calculatorと制限付きファイル読取
- [`types.h`](agent/types.md): Agent層の共有データ型

### Model

[model/README.md](model/README.md)を入口として、次の6ヘッダーを説明する。

- [`agent_dataset.h`](model/agent_dataset.md): 事前学習・SFT用windowとprovenance
- [`agent_tokenizer.h`](model/agent_tokenizer.md): SentencePieceと固定mode token
- [`agent_training.h`](model/agent_training.md): 学習、検証、checkpoint再開
- [`agent_transformer.h`](model/agent_transformer.md): decoder-only causal Transformer
- [`jawiki_sharding.h`](model/jawiki_sharding.md): Wikipedia JSONLのshard化
- [`model_language_model.h`](model/model_language_model.md): bundle、生成、Agent adapter

### NER

[ner/README.md](ner/README.md)を入口として、次の2ヘッダーを説明する。

- [`entity.h`](ner/entity.md): entity型、UTF-8 byte span、code point対応
- [`extractor.h`](ner/extractor.md): rule/model/hybrid抽出器

### 数値・CUDA基盤

`lib/`は`lib/include`の低レベル基盤を扱う。

- 基本型・実行基盤:
  [matrix](lib/matrix.md),
  [cuda_matrix](lib/cuda_matrix.md),
  [cuda_tensor](lib/cuda_tensor.md),
  [module](lib/module.md),
  [optimizer](lib/optimizer.md)
- CUDA資源管理:
  [cuda_memory](lib/cuda_memory.md),
  [cuda_bublas](lib/cuda_bublas.md),
  [cuda_function](lib/cuda_function.md)
- 演算:
  [Add](lib/cuda_function_add.md),
  [BatchMatMul](lib/cuda_function_batch_mat_mul.md),
  [IndexCrossEntropy](lib/cuda_function_index_cross_entropy.md),
  [LayerNorm](lib/cuda_function_layer_norm.md),
  [Linear](lib/cuda_function_linear.md),
  [Mask](lib/cuda_function_mask.md),
  [Permute](lib/cuda_function_permute.md),
  [Reshape](lib/cuda_function_reshape.md),
  [Scale](lib/cuda_function_scale.md),
  [Softmax](lib/cuda_function_softmax.md)

## 機能別の実装・学習・評価

| 分野 | 設計・手順 | 内容 |
|---|---|---|
| 内部要約 | [conversation_summary.md](conversation_summary.md) | 6 section形式、context予算、SFT変換、ablation |
| 日本語NER | [japanese_ner.md](japanese_ner.md) | データ取得・変換、bundle学習、評価、Agent統合 |
| 比較・判断 | [discussion_reasoning.md](discussion_reasoning.md) | 状態構造、決定的比較器、訂正、Reasoning経路 |
| 比較・判断評価 | [discussion_evaluation.md](discussion_evaluation.md) | scenario生成、CLI、SFT混合、導入前後評価 |
| 汎用trainer | [train/trainer.md](train/trainer.md) | template trainerの型契約と処理 |
| 会話モデル検証 | [validation/conversation_validation.md](validation/conversation_validation.md) | データ分割、GPU実測、保存・再読込 |

学習コマンドの実行順と現在利用するCLI例は、リポジトリ直下の
[README.md](../README.md)も参照する。

## 互換用文書と補助資料

- `aiagent_overview.md` と `aiagent_uml.md` は旧リンク互換用で、それぞれ
  [agent_overview.md](agent_overview.md) と [agent_uml.md](agent_uml.md) が正本。
- `mdn_1406dg.pdf` は補助資料であり、現在のC++公開APIの正本ではない。

## 更新規約

- ヘッダー別設計書は対応するヘッダーと同じstemを使う。
  例: `include/ai/agent/components.h` は `docs/agent/components.md`。
- 実測値には測定環境、条件、実行済みかどうかを併記する。
- smoke学習、本学習、未実行の予定を区別する。
- データライセンスと学習済みbundleの配布条件を同一視しない。
- 移動した文書は、既存リンクを壊さないため必要に応じてredirect文書を残す。
