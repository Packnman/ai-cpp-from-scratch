# UT-BRN-MOD-002 Pre-process 単体テスト仕様書


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
| UT-002-001 | Camera routing | CameraFrame | Object/Spatial recognizerへ配送 |
| UT-002-002 | Voice routing | Voice | Speech recognizerへ配送 |
| UT-002-003 | Text routing | Text | Context recognizerへ配送 |
| UT-002-004 | Object normalize | Fake object result | Perception生成 |
| UT-002-005 | Context command | 命令文 | Goal生成 |
| UT-002-006 | Context constraint | 制約文 | Constraint生成 |
| UT-002-007 | Low Confidence | 低confidence | uncertain扱い |
| UT-002-008 | Invalid Confidence | <0 or >1 | Reject |
| UT-002-009 | Recognition Failure | Fake failure | BrainError |
| UT-002-010 | External Timeout | timeout | BlockせずFallback/Failure |
| UT-002-011 | Fake Determinism | 同一入力N回 | 同一結果 |


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
