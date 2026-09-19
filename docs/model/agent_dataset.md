# `include/ai/model/agent_dataset.h`

## 目的

事前学習またはSFTのJSONLを、`AgentTransformer` が次token予測に使う `AgentWindow` 列へ変換する。データ由来情報とfingerprintも保持し、学習再開時の整合性確認に使う。

## 公開型

- `AgentWindow`: `inputs` と1位置先の `targets`、`mode`、元文書ID、有効token数を保持する。SFTではprompt部分のtargetを `pad_id` にしてloss対象外とする。
- `SftSourceProvenance`: 出典名、revision、converter versionの組。
- `AgentDataset`: 読み込み済みwindowとデータセットmetadataの所有者。

## 構築経路

`jawiki()` は各文書を `<s> text </s>` にし、最大contextに収まる連続窓へ分割する。`token_limit` は有効target token数の上限で、0は無制限である。

`sft()` は各JSONL行の `mode`、`prompt`、`output` を `<s> <MODE> prompt output </s>` にする。context超過行は切り詰めず `excluded_too_long()` に計上する。profileとsplitの混在を拒否し、profileごとのmode被覆も検査する。

## 順序と再現性

- `order(seed, epoch)` は同じ入力・seed・epochに対して決定的である。
- agent profileは `CHAT/PARSE/PLAN/EVALUATE/SUMMARIZE/MEMORY_WRITE/FINAL` を20/15/15/10/20/10/10の比率で100例に再標本化する。
- `fingerprint()` はmode、inputs、targetsから算出する。出典metadataそのもののfingerprintではない。
- `generate_sft_corpus()` は会話入力とローカルの決定的templateから、seed・split・出典を持つJSONLを生成する。

## エラーと制約

context 0、空データ、不明mode、不正profile構成は例外となる。長すぎるSFT例を暗黙にtruncateしない。`padding_ratio()` のbatch sizeは正でなければならない。

## 主な実装・検証先

- 実装: `src/model/agent_dataset.cpp`
- 学習入口: `src/model/agent_training.cpp`
- テスト: `tests/model/agent_model_check.cpp` のdataset・SFT関連case
