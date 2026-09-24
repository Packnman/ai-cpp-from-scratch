# UT-BRN-MOD-008 Planner 単体テスト仕様書


# 1. 目的

本書は対象モジュールの機能、状態遷移、異常処理、優先度、Timeout、Mock連携を単体で検証する。

# 2. テスト方針

- 外部SubsystemはMock / Fakeとする。
- 時刻依存処理はFake Clockで固定する。
- Randomを使用する場合は固定Seedとする。
- Unit TestはOSS Datasetへ依存しない。
- 同一入力に対して同一結果を返す決定論的Testを基本とする。

# 3. 主要テストケース

| ID | テスト | 入力 | 期待結果 |
| :- | :- | :- | :- |
| UT-008-001 | Simple Goal | 単純Goal | Action Plan生成 |
| UT-008-002 | Multi Action | 複合Goal | 順序付きPlan |
| UT-008-003 | Dependency | 依存Action | DAG生成 |
| UT-008-004 | Missing precondition | 前提不足 | prerequisite追加/Failure |
| UT-008-005 | Hard constraint | 違反 | Reject |
| UT-008-006 | Soft constraint | 違反 | score penalty |
| UT-008-007 | Resource conflict | 競合 | serial化/invalid |
| UT-008-008 | Invalid cycle | 循環依存 | Reject |
| UT-008-009 | Re-planning | 状態変化 | 新Plan |
| UT-008-010 | External timeout | timeout | RuleBased fallback |
| UT-008-011 | Determinism | 同一Context | 同一Plan |


# 4. 異常・境界試験

- 空入力
- 最大件数
- 閾値直前 / 直後
- 重複入力
- 順序逆転
- Timeout
- Mock Failure

# 5. 合格条件

- 必須Test Caseが全件PASS
- Safety / Constraint関連の禁止条件違反0件
- 未処理例外0件
- Memory Leak / Data Raceを発生させない
