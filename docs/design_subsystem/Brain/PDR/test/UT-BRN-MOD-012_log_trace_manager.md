# UT-BRN-MOD-012 Log / Trace Manager 単体テスト仕様書


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
| UT-012-001 | Trace write | TraceEvent | Backend記録 |
| UT-012-002 | Goal correlation | GoalId | 保持 |
| UT-012-003 | Plan correlation | PlanId | 保持 |
| UT-012-004 | Action correlation | ActionId | 保持 |
| UT-012-005 | Error report | BrainError | 保存 |
| UT-012-006 | Version info | versions | 記録 |
| UT-012-007 | Backend failure | FailingBackend | Brain継続 |
| UT-012-008 | Queue overflow | 大量Log | Low priorityからDrop |
| UT-012-009 | Critical preserve | Critical Log | 優先保存 |
| UT-012-010 | Replay metadata | 一連のTrace | 必須情報保持 |


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
