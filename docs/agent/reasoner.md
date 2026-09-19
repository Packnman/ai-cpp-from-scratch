# `include/ai/agent/reasoner.h` 設計書

## 目的

`IReasoner`の3 backendを定義し、決定処理と文章生成の配分を切り替える。

## Backend

| 型 | 役割 |
|---|---|
| `RuleReasoner` | command解析、rule計画・評価・最終整形、動作確認会話 |
| `ModelReasoner` | mode別の構造化生成、会話、要約、記憶候補 |
| `HybridReasoner` | parse/plan/evaluate/final/memoryをrule、chatとsummaryをmodel |

明示的なcomparisonは外側のDefault componentが決定経路を作る。`ModelReasoner`でも
comparisonの最終文は検証済みdecisionをrule形式で整形し、モデルに書き換えさせない。

## 構造化生成

`ModelReasoner::structured`はmodeごとのJSON schemaを要求し、parseまたはschema検証に
失敗した場合だけ1回repairする。生成できたことと内容の正しさは区別する。

## 要約

`validate_structured_summary`はFACTS、DECISIONS、CONSTRAINTS、OPEN_QUESTIONS、
ACTIVE_TASKS、CORRECTIONSの順序・非空本文・token上限を検査する。更新失敗時はAgentが
以前のsummaryと未要約履歴を保持する。

## 記憶候補

ModelReasonerはNERを未確定候補として渡し、否定・提案・訂正の解決を指示する。最終的な
保存可否と閾値は`Agent`が決める。Hybrid/Ruleでは明示`/remember`だけを候補化する。

## 実装・検証

- 実装: `src/agent/reasoner.cpp`
- テスト: `tests/agent/agent_check.cpp`, `tests/model/agent_model_check.cpp`
