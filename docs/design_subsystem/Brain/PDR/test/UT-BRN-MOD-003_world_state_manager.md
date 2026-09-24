# UT-BRN-MOD-003 World State Manager 単体テスト仕様書


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
| UT-003-001 | New Entity | 新規Perception | 追加 |
| UT-003-002 | Same ID update | 同一ID | 更新 |
| UT-003-003 | Tracking統合 | 同一TrackingID | 重複生成しない |
| UT-003-004 | Condition update | Condition変化 | value更新 |
| UT-003-005 | Safety State | Safety更新 | 高優先で反映 |
| UT-003-006 | Stale transition | TTL超過 | stale扱い |
| UT-003-007 | Conflict | 矛盾状態 | conflict検出 |
| UT-003-008 | Snapshot immutable | snapshot後update | snapshot不変 |
| UT-003-009 | Version increment | update | version増加 |
| UT-003-010 | Concurrent read/write | 並行アクセス | raceなし |


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
