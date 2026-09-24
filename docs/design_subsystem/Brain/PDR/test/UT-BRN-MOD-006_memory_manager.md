# UT-BRN-MOD-006 Memory Manager 単体テスト仕様書


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
| UT-006-001 | STM insert | MemoryItem | RingBuffer追加 |
| UT-006-002 | STM overflow | 容量超過 | oldest eviction |
| UT-006-003 | LTM insert | MemoryItem | SQLite保存 |
| UT-006-004 | Update/Delete | 既存ID | 正しく更新/削除 |
| UT-006-005 | Search by type | type指定 | 該当のみ |
| UT-006-006 | Search by tag | tag指定 | 該当のみ |
| UT-006-007 | Ranking | 複数候補 | 仕様score順 |
| UT-006-008 | Transaction rollback | 途中失敗 | 部分更新なし |
| UT-006-009 | DB unavailable | DB failure | fallback |
| UT-006-010 | STM-only fallback | DB unusable | Brain継続 |
| UT-006-011 | Promote STM→LTM | 昇格条件成立 | LTM保存 |


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
