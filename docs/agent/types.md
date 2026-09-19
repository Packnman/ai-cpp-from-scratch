# `include/ai/agent/types.h` 設計書

## 目的

Agent制御層でcomponent間を流れる値型を一箇所に定義する。型は処理状態を表すが、
それ自体で内容の真偽やschema妥当性を保証しない。

## 制御型

- `RequestType`: 会話、質問、記憶検索、複雑Reasoning、tool、robot分類。
- `TaskType`: ToolまたはReasoning。
- `ToolStatus`: Success、RetryableError、PermanentError、NeedsReplan。
- `EvaluationStatus`: Success、Retry、Replan、Failed。
- `Task`/`Plan`: operation、JSON引数、依存IDを持つDAG表現。
- `ToolResult`/`EvaluationResult`: 実行状態と評価状態を分離する。

## 入力・記憶・応答型

- `ParsedInput`: 原文、intent、goal、constraint、JSON引数、未確定NER候補。
- `Constraint.critical`: contextから省略してはならない制約。
- `MemoryCandidate`: 永続化前候補。`MemoryRecord`はIDとtimestampを追加する。
- `ConversationTurn`: user/assistant本文と、その発話のNER候補。
- `AgentState`: 1ターンの解析、Plan、結果、履歴、summary、NER、議論状態のsnapshot。
- `AgentResponse`: success、表示文、state、error。

## 互換性と注意

`arguments`、`ToolResult.value`、`discussion_state`は拡張可能なJSON境界である。利用側は
operation固有schemaを検証する。NER候補は事実ではなく、scoreも確率ではない。

`to_string`はRequestType/MemoryTypeの永続・表示名、`memory_type_from_string`はDB/tool
入力の逆変換に使う。enum順序へ依存する箇所があるため、既存値の途中挿入は避ける。

## 実装・検証

- 変換実装: `src/agent/components.cpp`
- 利用テスト: `tests/agent/agent_check.cpp`, `tests/agent/control_flow_check.cpp`
