# UT-BRN-MOD-009 Execution Manager 単体テスト仕様書


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
| UT-009-001 | Sequential | 依存Plan | 順番通りDispatch |
| UT-009-002 | Parallel | 非競合 | 同時Running |
| UT-009-003 | Exclusive conflict | 同一Resource | 同時Dispatchしない |
| UT-009-004 | Shared resource | Shared | 許可条件で同時利用 |
| UT-009-005 | Success | 成功Result | Succeeded |
| UT-009-006 | Failure | 失敗Result | Failed + replan |
| UT-009-007 | Timeout | 無応答 | Cancel→Timeout |
| UT-009-008 | Cancel running | Cancel | 停止確認後release |
| UT-009-009 | EmergencyStop | EStop | 新規Dispatch禁止 |
| UT-009-010 | Duplicate result | 同一Result再送 | 二重遷移しない |


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
