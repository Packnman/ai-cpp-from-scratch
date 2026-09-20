# Simulation Requirement

## Scope

全身ロボットの剛体運動、関節状態、接触、センサ真値を再現し、Control→Actuator→Plant の閉ループ検証を headless と viewer の双方で可能にする。

## Functional requirements

| ID | Requirement | Verification |
|---|---|---|
| SIM-REQ-001 | Control/Actuator は MuJoCo 型を公開 API に含めない | dependency review |
| SIM-REQ-002 | Plant は initialize/reset/step/torque/force/constraint/state を提供する | UT-SIM-MOD-002 |
| SIM-REQ-003 | logical ID と MuJoCo ID/address を初期化時に対応付ける | UT-SIM-MOD-002 |
| SIM-REQ-004 | 物理・Actuator・Control の周期を独立管理する | UT-SIM-MOD-001 |
| SIM-REQ-005 | 関節、剛体、接触、センサ状態を返す | UT-SIM-MOD-005/006 |
| SIM-REQ-006 | Actuator 出力を関節トルク、直動力、拘束へ変換する | UT-SIM-MOD-004 |
| SIM-REQ-007 | headless 実行は同一入力で決定的である | IT-SIM-001 |
| SIM-REQ-008 | 単関節の符号・重力・解析解を検証する | IT-SIM-001 |
| SIM-REQ-009 | Actuator および Control を含む閉ループを検証する | IT-SIM-002/003 |
| SIM-REQ-010 | 段階 MJCF と include 分割全身 MJCF を提供する | CLI smoke test |

## Non-functional requirements

- C++20、MuJoCo C API、RAII を使用する。
- シミュレーション時刻は壁時計から分離し `mjData::time` を正とする。
- 既定の physics/actuator/control timestep は 1/1/10 ms とする。
- MuJoCo 3.13.0 を固定し、バージョン変更はテスト結果と共にレビューする。
- ASan/UBSan を CMake オプションで有効化できること。
