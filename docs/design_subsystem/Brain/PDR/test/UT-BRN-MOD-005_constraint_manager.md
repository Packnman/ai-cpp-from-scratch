# UT-BRN-MOD-005 Constraint Manager 単体テスト仕様書


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
| UT-005-001 | Add Hard | critical=true | active追加 |
| UT-005-002 | Add Soft | critical=false | active追加 |
| UT-005-003 | Hard violation | 違反Plan | Plan invalid |
| UT-005-004 | Soft violation | 違反Plan | valid + penalty |
| UT-005-005 | Scope | Goal/Action scope | 対象範囲のみ |
| UT-005-006 | Expiration | 期限超過 | inactive |
| UT-005-007 | Safety conflict | Safety vs Policy | Safety優先 |
| UT-005-008 | Unresolvable conflict | 解決不能 | PlanningFailure |
| UT-005-009 | Replan event | Safety追加 | ReplanRequired |


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
