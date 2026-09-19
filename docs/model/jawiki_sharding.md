# `include/ai/model/jawiki_sharding.h`

## 目的

巨大なWikipedia JSONLを、学習時に逐次処理できるshard群とmanifestへ変換する。

## `shard_jawiki()`

`source` の各行をJSONとして読み、必須の `id` と `text` を検証して `output` 配下へ書く。`target_bytes` はshardサイズの目標値であり、JSONL行を途中で切らないためsoft limitである。単一行が上限を超える場合も、その行は完全な形で1 shardに残る。

## 完全性と再実行

- shardは一時名へ書いてから確定名へ移し、不完全ファイルを完成品として見せない。
- manifestには順序、byte数、行数、fingerprintなど、後続学習が入力を検証するための情報を記録する。
- 既存出力はmanifestと実ファイルが整合するとき再利用し、不整合な出力を黙って上書きしない。
- 元JSONL本文をGit管理するためのAPIではない。

## 主な実装・検証先

- 実装: `src/model/jawiki_sharding.cpp`
- 利用側: `pretrain_sharded()`（[agent_training.md](agent_training.md)）
- テスト: Jawiki sharding/manifest関連check
