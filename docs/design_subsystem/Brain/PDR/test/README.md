# Brain System テスト仕様書

Brain System モジュール詳細設計に対する単体テスト仕様書、およびBrain単体結合テスト仕様書を管理する。

# 1. モジュール単体テスト

| ID | Module | Test Spec |
| :- | :- | :- |
| BRN-MOD-001 | Input Adapter | [UT-BRN-MOD-001](./UT-BRN-MOD-001_input_adapter.md) |
| BRN-MOD-002 | Pre-process | [UT-BRN-MOD-002](./UT-BRN-MOD-002_preprocess.md) |
| BRN-MOD-003 | World State Manager | [UT-BRN-MOD-003](./UT-BRN-MOD-003_world_state_manager.md) |
| BRN-MOD-004 | Goal Manager | [UT-BRN-MOD-004](./UT-BRN-MOD-004_goal_manager.md) |
| BRN-MOD-005 | Constraint Manager | [UT-BRN-MOD-005](./UT-BRN-MOD-005_constraint_manager.md) |
| BRN-MOD-006 | Memory Manager | [UT-BRN-MOD-006](./UT-BRN-MOD-006_memory_manager.md) |
| BRN-MOD-007 | Policy Manager | [UT-BRN-MOD-007](./UT-BRN-MOD-007_policy_manager.md) |
| BRN-MOD-008 | Planner | [UT-BRN-MOD-008](./UT-BRN-MOD-008_planner.md) |
| BRN-MOD-009 | Execution Manager | [UT-BRN-MOD-009](./UT-BRN-MOD-009_execution_manager.md) |
| BRN-MOD-010 | Post-process | [UT-BRN-MOD-010](./UT-BRN-MOD-010_post_process.md) |
| BRN-MOD-011 | External AI Adapter | [UT-BRN-MOD-011](./UT-BRN-MOD-011_external_ai_adapter.md) |
| BRN-MOD-012 | Log / Trace Manager | [UT-BRN-MOD-012](./UT-BRN-MOD-012_log_trace_manager.md) |

# 2. Brain単体結合テスト

[IT-BRAIN Brain System 単体結合テスト仕様書](./IT-BRAIN_brain_integration_test.md)

# 3. テスト階層

```text
Unit Test
    ↓
Brain Integration Test
    ↓
Regression Test
    ↓
OSS Dataset Benchmark
    ↓
Simulator / Robot Integration
```

Unit TestとBrain Integration Testは決定論的なFake / Mock / Synthetic Dataを基本とする。
OSS Datasetは性能評価・Benchmark層として使用し、Core機能の単体合否を外部Dataset依存にしない。

# 4. 実装と実行

各仕様書に対応するテスト実装は `tests/brain/ut_brn_mod_*.cpp`、結合テストは
`tests/brain/it_brain_integration.cpp` に配置する。仕様上の機能が現行APIに存在しない場合は、
テストを削除したり成功扱いにしたりせず、理由付きの `SKIP` として出力する。

対応状況は [implementation_coverage.md](./implementation_coverage.md) を参照する。

```sh
cmake -S . -B build -DBUILD_TESTING=ON \
  -DAI_CPP_BUILD_CUDA_LIB=OFF -DAI_CPP_BUILD_MODEL=OFF
cmake --build build -j2
ctest --test-dir build -L spec --output-on-failure
ctest --test-dir build -L spec -V
```

最後のコマンドは仕様書ごとのPASS/SKIP内訳も表示する。
