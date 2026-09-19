# `include/ai/model/agent_training.h`

## 目的

`AgentTransformer` の評価、通常学習、Wikipedia shard単位の事前学習を提供する。中断再開でモデルだけでなくoptimizer、進捗、乱数に関係する状態の整合性を守る。

## 設定

`TrainingConfig` はstep、batch、勾配蓄積、token budget、学習率schedule、VRAM上限、gradient clipping、seedを持つ。`validate()` はゼロ値や範囲、scheduleの矛盾を実行前に拒否する。

`ShardedTrainingConfig` は共通学習設定にepoch数と1回に処理する最大shard数を加える。`max_shards` は巨大corpusを明示的に区切って再開するための上限である。

## 操作

- `validate()`: training状態を変更せず、token当たりloss、perplexity、throughput、padding率、bits/byteを集計する。
- `train()`: train/validation fingerprintとtokenizerを紐付けて学習し、bundleとcheckpointを保存する。
- `pretrain_sharded()`: manifest順にshardを検証・学習し、epoch/shard境界から再開する。

## 再開契約

`resume=true` は「似た設定から続ける」指定ではない。保存済みのモデル構成、tokenizer、データfingerprint、学習設定と互換な状態だけを再開する。勾配蓄積途中を完了済みupdateとして扱わず、shard manifestの改変も拒否する。

## 指標の解釈

`ValidationMetrics` はlanguage-model lossであり、Agentの判断正解率やNER F1ではない。smoke学習の完走も本学習の品質達成を意味しない。

## 主な実装・検証先

- 実装: `src/model/agent_training.cpp`
- 永続化: [model_language_model.md](model_language_model.md)
- テスト: training/checkpoint/sharded-training関連check
