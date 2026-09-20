# UT-BRN-MOD-004 Goal Manager 単体テスト仕様書


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
| UT-004-001 | Add Goal | 新規Goal | Pending登録 |
| UT-004-002 | Highest Priority | 複数Goal | 最高PriorityがActive |
| UT-004-003 | Safety Goal | Safety由来Goal | 通常Goalより優先 |
| UT-004-004 | Preemption | 高Priority追加 | Current Suspended |
| UT-004-005 | Completion | 完了条件成立 | Achieved |
| UT-004-006 | Failure | 失敗入力 | Failed |
| UT-004-007 | Cancel Active | Active cancel | Cancelled + cancel event |
| UT-004-008 | Parent/Sub Goal | 親子Goal | 関係保持 |
| UT-004-009 | Duplicate ID | 同一ID | Reject |


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
