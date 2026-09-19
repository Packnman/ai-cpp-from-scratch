# `include/ai/agent/components.h` 設計書

## 目的

Agentの各段階をinterfaceで分離し、rule/model backend、tool、記憶、限定Reasoningを
差し替え可能にする。`Default*`群は標準の配線と安全境界を実装する。

## Interface

| 型 | 責務 |
|---|---|
| `IInputParser` | 原文を`ParsedInput`へ変換 |
| `IRouter` | `RequestType`を決定 |
| `IPlanner` | Plan作成と再計画 |
| `ITool` / `ToolRegistry` | operation名から外部・決定的操作を実行 |
| `IExecutor` | Tool/Reasoning taskを統一実行 |
| `IReasoningTaskExecutor` | Toolでない限定Reasoningを実処理 |
| `IEvaluator` | retry/replan/failure/success判定 |
| `IAggregator` | 成功結果の集約 |
| `IMemoryManager` | 長期記憶の保存・検索 |
| `ILanguageModel` | mode付き文字列補完とtoken計数 |
| `IReasoner` | parse/plan/evaluate/chat/summary/final等のbackend契約 |

## 既定実装の規則

- `/compare JSON`はmodel parseを迂回し、`discussion.compare`の単一Planになる。
- Tool taskは`ToolRegistry`、Reasoning taskは`IReasoningTaskExecutor`へ送る。
- 未登録tool、未接続Reasoning、未知operationを成功として扱わない。
- `memory acknowledgement`だけは、後段の記憶保存を許可する明示的内部操作。
- AggregatorはJSON値の違いだけで矛盾とせず、明示された`semantic_conflicts`のみ集める。
- Planは最大32 task。task IDは非空・一意、依存先が存在し、DAGでなければならない。

## ModelMode予算

contextは1024 token、BOSとmode tokenに2 tokenを予約する。

- `Evaluate`/`MemoryQuery`: 出力128、prompt 894。
- その他: 出力256、prompt 766。

`ILanguageModel::token_count`の既定はUTF-8 code point概算で、実モデルはtokenizerによる
正確な計数へoverrideする。

## 実装・検証

- 実装: `src/agent/components.cpp`
- テスト: `tests/agent/control_flow_check.cpp`, `tests/agent/agent_check.cpp`
