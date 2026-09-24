# UT-BRN-MOD-011 External AI Adapter 単体テスト仕様書


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
| UT-011-001 | Normal response | 正常Fake | Succeeded |
| UT-011-002 | Request ID mismatch | 誤ID | Reject |
| UT-011-003 | Schema mismatch | 誤version | Reject |
| UT-011-004 | Timeout | 無応答 | Timeout |
| UT-011-005 | Late response | timeout後応答 | 現Contextへ反映しない |
| UT-011-006 | Retry idempotent | 一時失敗 | Retry |
| UT-011-007 | Non-idempotent | 一時失敗 | 自動Retryしない |
| UT-011-008 | Goal changed | 古いResponse | Reject |
| UT-011-009 | Local fallback | 外部失敗 | RuleBased等へ |
| UT-011-010 | Fake determinism | 同一Request | 同一Response |


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
