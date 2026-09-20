# UT-BRN-MOD-007 Policy Manager 単体テスト仕様書


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
| UT-007-001 | Add Policy | Policy | 登録成功 |
| UT-007-002 | Applicable true | 成立context | 候補含む |
| UT-007-003 | Applicable false | 不成立context | 候補外 |
| UT-007-004 | Priority | 複数候補 | 高priority優先 |
| UT-007-005 | Success update | 成功結果 | successRate更新 |
| UT-007-006 | Failure update | 失敗結果 | successRate低下 |
| UT-007-007 | Version increment | 更新 | version増加 |
| UT-007-008 | Critical violation | Constraint違反 | 候補除外 |


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
