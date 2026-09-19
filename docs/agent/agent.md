# `include/ai/agent/agent.h` 設計書

## 目的

`Agent`は入力解析から応答返却までを統括するアプリケーションサービスである。解析、
計画、実行、評価、集約、記憶、生成の具体方式は注入されたinterfaceへ委譲する。

## 公開契約

- constructorは8個の必須componentと任意の`IEntityExtractor`を受け取る。必須componentが
  nullなら`invalid_argument`。
- `process(input)`は1ターンを同期実行し、成功・応答・観測可能な`AgentState`を返す。
- 内部例外は`AgentResponse.success=false`と`error`へ変換する。

## 処理順

1. parse前の原文へ任意NERを適用し、発話ID付き候補を`ParsedInput`へ設定する。
2. routeし、SQLite記憶を最大8件取得する。
3. 通常会話では必要時に履歴を要約して`chat`を呼ぶ。
4. その他ではPlan検証、依存順実行、最大2 retry、最大3 replanを行う。
5. 集約後、限定比較の`discussion_state`をAgent内のcanonical stateとして保持する。
6. 応答後に記憶候補を閾値検査し、短期履歴を更新する。

## 状態と不変条件

- `_recent`は要約前の会話、`_summary`は検証済み構造化要約。
- `_discussion_state`は要約から独立し、要約失敗で上書きされない。
- comparison入力の巨大JSONは履歴へ複製せずscenario参照だけを残す。
- comparison結果は通常の長期記憶へ無条件保存しない。
- インスタンスは可変状態を持つため、同一インスタンスの並行`process`は想定しない。

## 実装・検証

- 実装: `src/agent/agent.cpp`
- 主テスト: `tests/agent/agent_check.cpp`, `tests/agent/discussion_agent_check.cpp`
- 関連: [components.md](components.md), [types.md](types.md),
  [discussion.md](discussion.md)
