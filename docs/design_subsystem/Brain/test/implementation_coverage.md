# Brainテスト仕様 実装対応状況

更新日: 2026-09-20

## 対応方針

- 仕様書ごとに独立した実行ファイルを作る。
- 既存APIで検証可能な項目は決定論的なテストとして実行する。
- 現行設計にAPIや状態表現がない項目は、未実装理由を伴う `SKIP` として残す。
- `SKIP` は合格件数に含めない。CTestのプロセス成功は、仕様項目の全実装を意味しない。
- 既存の `brain_phase*_check` はフェーズ単位の回帰テストとして併存させる。

## 仕様書と実装

| 仕様書 | テスト実装 | 現在の実行結果 |
| :- | :- | :- |
| UT-BRN-MOD-001 Input Adapter | `tests/brain/ut_brn_mod_001_input_adapter.cpp` | 13 PASS / 0 SKIP |
| UT-BRN-MOD-002 Pre-process | `tests/brain/ut_brn_mod_002_preprocess.cpp` | 4 PASS / 3 SKIP群 |
| UT-BRN-MOD-003 World State Manager | `tests/brain/ut_brn_mod_003_world_state_manager.cpp` | 8 PASS / 2 SKIP群 |
| UT-BRN-MOD-004 Goal Manager | `tests/brain/ut_brn_mod_004_goal_manager.cpp` | 9 PASS / 0 SKIP |
| UT-BRN-MOD-005 Constraint Manager | `tests/brain/ut_brn_mod_005_constraint_manager.cpp` | 6 PASS / 2 SKIP群 |
| UT-BRN-MOD-006 Memory Manager | `tests/brain/ut_brn_mod_006_memory_manager.cpp` | 8 PASS / 2 SKIP群 |
| UT-BRN-MOD-007 Policy Manager | `tests/brain/ut_brn_mod_007_policy_manager.cpp` | 7 PASS / 1 SKIP群 |
| UT-BRN-MOD-008 Planner | `tests/brain/ut_brn_mod_008_planner.cpp` | 11 PASS / 0 SKIP |
| UT-BRN-MOD-009 Execution Manager | `tests/brain/ut_brn_mod_009_execution_manager.cpp` | 10 PASS / 0 SKIP |
| UT-BRN-MOD-010 Post-process | `tests/brain/ut_brn_mod_010_post_process.cpp` | 5 PASS / 3 SKIP群 |
| UT-BRN-MOD-011 External AI Adapter | `tests/brain/ut_brn_mod_011_external_ai_adapter.cpp` | 7 PASS / 2 SKIP群 |
| UT-BRN-MOD-012 Log / Trace Manager | `tests/brain/ut_brn_mod_012_log_trace_manager.cpp` | 8 PASS / 1 SKIP群 |
| IT-BRAIN | `tests/brain/it_brain_integration.cpp` | 6 PASS / 8 SKIP群 |

`SKIP群` は1行に複数の仕様IDをまとめて表示する場合があるため、仕様ケース数ではなく
テスト出力上のSKIPレコード数を示す。

## 主な未実装領域

- 認識器の信頼度・失敗・タイムアウトを注入するテストダブル
- World Stateの追跡ID解決と競合メタデータ
- 制約間競合の解決とreplanイベント
- Memoryのトランザクション障害注入とSTM自動昇格
- Policy ManagerとConstraint Managerの直接連携
- External AIの再試行、およびBrainSystemへの統合
- 優先度付き有界ログキューとBrainSystemからのtrace参照
- BrainSystem結合レベルでの障害注入、実行中E-Stop、live replan

## 実測結果

CPU構成で `ctest -L spec` を実行し、13実行ファイル中13件が成功した。
テスト内訳は合計102 PASS、24 SKIPレコードである。これは実装済み範囲の回帰結果であり、
SKIP対象の仕様適合を示すものではない。
