# 限定比較・判断の実行、学習、評価

## 実装対応

| 要求概念 | 実装 |
|---|---|
| `DiscussionState` / `Claim` / `Evidence` / `Decision` | `include/ai/agent/discussion.h` |
| `StateUpdate` | 同headerの型、および `DiscussionEngine::execute` の検証後適用 |
| 数値制約、欠損、衝突、優先順位 | `src/agent/discussion.cpp` の `decide` |
| Reasoning実行 | `IReasoningTaskExecutor` と `DiscussionEngine` |
| 形式検証 | `validate_discussion_result` |
| Agent統合 | `DefaultInputParser` / `DefaultPlanner` / `DefaultEvaluator` / `Agent::process` |
| scenario生成・分割 | `scripts/generate_discussion_data.py` |
| 評価 | `discussion_cli evaluate` |
| 既存SFTとの混合 | `scripts/mix_discussion_sft.py` |

`Evidence` の `start` と `end` は出典原文に対するUTF-8 byte offsetの半開区間である。
資料由来であることは資料がその内容を記述したという意味に限り、世界での真実性を
保証しない。user/model由来の未検証Claim、proposed/retracted/superseded Claimは比較の
事実へ昇格しない。constraintはユーザーの選択条件なのでuser由来でも適用する。

衝突はsubject、attribute、time、condition、unitがすべて等しい利用可能Claim間の値の
不一致だけである。対象や時点の違い、提案、明示的訂正は衝突にしない。訂正はupdate
IDで冪等化し、旧値、旧根拠ID、新根拠IDをrevisionへ残す。状態をコピーして検証する
ため、不正なupdateで一部だけ更新されない。

## 再現コマンド

```sh
cmake -S . -B build/discussion -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/discussion -j2
ctest --test-dir build/discussion --output-on-failure

./scripts/generate_discussion_data.py data/discussion-reasoning \
  --seed 20260918 --groups-per-category 10
build/discussion/discussion_cli evaluate \
  data/discussion-reasoning/validation.jsonl
build/discussion/discussion_cli evaluate \
  data/discussion-reasoning/test.jsonl
```

データ本文は`data/`配下なのでGit対象外である。manifestにはseed、scenario group単位の
split方式、件数、各splitのSHA-256を保存する。各カテゴリ内のgroupを70/10/20へ
決定的に順位付けし、訂正前後と言い換えvariantを同一splitへ置く。testは実装調整に
使用しない。

単独実行は次の形式である。CLI Agentでは同じJSONを1行の`/compare JSON`として渡す。

```sh
build/discussion/discussion_cli run scenario.json
build/discussion/agent_cli --backend rule --memory-db /tmp/agent.sqlite3
```

## SFTへの追加

生成物の`*.sft.jsonl`は既存mode IDの`FINAL`追加例であり、単独のagent profileでは
ない。既存の全mode SFTへ混ぜ、別bundleへ保存する。

```sh
./scripts/mix_discussion_sft.py \
  --base data/agent-v3-base-sft/train.jsonl \
  --discussion data/discussion-reasoning/train.sft.jsonl \
  --output data/agent-v3-discussion-sft/train.jsonl --seed 20260918
./scripts/mix_discussion_sft.py \
  --base data/agent-v3-base-sft/validation.jsonl \
  --discussion data/discussion-reasoning/validation.sft.jsonl \
  --output data/agent-v3-discussion-sft/validation.jsonl --seed 20260918

# smoke（既存pretrain bundleを変更せず別出力へ1 update）
build/model/agent_model_cli sft \
  --train data/agent-v3-discussion-sft/train.jsonl \
  --validation data/agent-v3-discussion-sft/validation.jsonl \
  --source models/agent_v3_pretrain_jawiki \
  --output models/agent_v3_discussion_smoke --steps 1 --batch-size 1 \
  --accumulation 1 --seed 20260918

# 本学習例
build/model/agent_model_cli sft \
  --train data/agent-v3-discussion-sft/train.jsonl \
  --validation data/agent-v3-discussion-sft/validation.jsonl \
  --source models/agent_v3_pretrain_jawiki \
  --output models/agent_v3_discussion_sft --steps 50000 --batch-size 1 \
  --accumulation 4 --seed 20260918
```

既存bundleはこの追加例を学習していない。実行時の`/compare`は未学習モデルによる
自己評価を通さず、C++の決定的経路で完結する。SFTは将来の自然な構造化入出力改善用
であり、学習済みと表示するには別bundleの学習・held-out評価が必要である。

## 2026-09-18 実測

Debug CPU build、Linux、合成seed 20260918で測定した。validationは20 scenario / 22
turn、testは40 scenario / 44 turnで、10カテゴリをどちらにも含む。

| 指標 | legacy相当 | validation | test |
|---|---:|---:|---:|
| structured decision正解率 | 0.000 | 1.000 | 1.000 |
| 該当なし・不足・衝突の正解率 | 未対応 | 1.000 | 1.000 |
| 根拠ID完全一致 | 未対応 | 1.000 | 1.000 |
| 訂正後の最終判断 | 未対応 | 1.000 | 1.000 |
| 別対象・別時点の誤衝突率 | 未対応 | 0.000 | 0.000 |
| 資料外Claim追加率 | 未対応 | 0.000 | 0.000 |
| schema妥当率 | 0.000 | 1.000 | 1.000 |
| throughput | - | 478 turn/s | 502 turn/s |
| process peak RSS | - | 4,128 KiB | 4,192 KiB |
| additional runtime model | - | 0 byte | 0 byte |
| mean serialized state | - | 3,146 byte | 3,146 byte |

全入力JSONは766 codepointを超えるが、比較器はモデルcontextへ投入しない。Agent履歴へは
巨大JSONではなくscenario参照だけを残し、ユーザー応答もdecisionの短い結論・根拠ID・
未確認事項だけである。SFT promptは必要表だけに圧縮し、生成テストで最大766 codepoint
以下（今回214–734）を検証する。codepointは厳密なSentencePiece token数ではないため、
実学習時には`AgentDataset`の1024-token除外統計も確認する。

## 未実行・既知の限界

- GPUはRTX 3060 Ti 8 GiBだが測定時の空きが約2.97 GiBで、既存pipelineの4 GiB安全閾値
  を満たさなかった。smoke/full SFTは未実行で、loss、勾配、再ロード後の生成精度は
  報告しない。上記smokeコマンドから再開する。
- 自然文からClaimやconstraintへの一般変換は対象外で、初期入力は明示JSONである。
  NER候補は事実に昇格せず、属性対応と訂正認識はJSON更新として分離した。
- DiscussionStateはprocess内メモリのみで、再起動後の永続化は未実装。SQLite schemaを
  変更していないため既存DBとの互換性は維持する。
- 世界の真偽、単位変換、曖昧な時点、一般自然文の含意は解かない。比較は`<=`、`>=`、
  `==`、単位完全一致、current/defaultに限定する。
- 合成課題での1.0は決定規則とfixtureの整合を示すだけで、自由討論能力や実会話での
  汎化を示さない。通常会話、tool、NER、memoryのCPU回帰はCTestで確認する。
