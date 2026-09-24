# UT-BRN-MOD-010 Post-process 単体テスト仕様書


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
| UT-010-001 | Raw result preserve | ActionResult | 評価前Log |
| UT-010-002 | Success | 成功 | Memoryへ成功経験 |
| UT-010-003 | Failure | 失敗 | 失敗理由保存 |
| UT-010-004 | Goal complete | 完了条件成立 | Achieved |
| UT-010-005 | LTM candidate | 新Object | LTM候補 |
| UT-010-006 | Policy update | 成功/失敗 | 統計更新 |
| UT-010-007 | Learning sample | ActionResult | sample生成 |
| UT-010-008 | Failure→replan | 失敗 | Replan event |
| UT-010-009 | Evaluation exception | 例外 | Raw Result消失なし |


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
