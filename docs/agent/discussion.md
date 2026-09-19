# `include/ai/agent/discussion.h` 設計書

## 目的

提供資料に限定した候補比較、必須制約の検査、根拠不足・衝突の整理、訂正後の再判定を
生成モデルから独立して実行する。

## データモデル

- `Evidence`: 根拠本文、source ID、UTF-8 byte半開区間、origin、内容検証済みflag。
- `Claim`: speaker、subject、attribute、time、condition、数値、unit、stance、根拠ID。
- `DiscussionConstraint`: `<=`/`>=`/`==`の数値条件と必須flag。
- `Decision`: 4 status、短い結論、任意の選択肢、使用根拠、未確認事項。
- `DiscussionState`: scenario単位の全状態、適用済みupdate ID、revision、現判断。

## 信頼規則

Evidence originはuser/document/tool/model/verified judgementを区別する。document由来は
「資料に記載された」ことの根拠で、世界の真実性保証ではない。未検証user/model、
proposed/retracted/superseded Claimは比較用factとして使わない。

衝突はsubject、attribute、time、condition、unitが同一で値だけが異なる場合に限定する。
別対象・別時点・提案・明示訂正を誤って衝突扱いしない。

## 更新と出力

対応updateは`correct_constraint`、`correct_claim`、`set_priorities`、`retract_claim`。
update IDで冪等化し、旧値・旧根拠をrevisionへ保存する。コピー上で検証してからcommit
するため、不正更新は部分反映されない。

`validate_discussion_result`はstatus、必須文字列、配列型、supported時の選択肢、未知の
Evidence IDを検査する。形式妥当性と意味正解は別評価である。

## ライフサイクル

`DiscussionEngine`はscenario IDごとの状態をprocess内mapへ保持する。永続化と並行実行は
未対応。未知Reasoning operationはPermanentError。

## 実装・検証

- 実装: `src/agent/discussion.cpp`
- テスト: `tests/agent/discussion_check.cpp`, `tests/agent/discussion_agent_check.cpp`
- 評価: `train/discussion_main.cpp`
