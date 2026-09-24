# UT-BRN-MOD-001 Input Adapter 単体テスト仕様書


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
| UT-001-001 | Sensor正常入力 | Camera message | Sensor / CameraFrameとしてBrainInput生成 |
| UT-001-002 | HMI正常入力 | Text message | HumanInterface / Textへ変換 |
| UT-001-003 | Control正常入力 | ActionResult | Control / ActionResultへ変換 |
| UT-001-004 | Safety正常入力 | EmergencyStop相当 | Safety Queueへ投入 |
| UT-001-005 | Unknown Source | 不正Source | Reject |
| UT-001-006 | Unknown Type | 不正Type | Reject |
| UT-001-007 | Schema不一致 | unsupported version | Reject + Error Log |
| UT-001-008 | Missing Field | timestamp欠落 | Reject |
| UT-001-009 | Stale Sensor | 古いtimestamp | Stale判定 |
| UT-001-010 | Safety Priority | Safety + Sensor同時投入 | Safetyを先にpoll |
| UT-001-011 | FIFO | 同Priority複数件 | 投入順 |
| UT-001-012 | Queue Overflow | 上限超過 | Drop Policy |
| UT-001-013 | Concurrent Push | 複数Thread | 欠損・破損なし |


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
