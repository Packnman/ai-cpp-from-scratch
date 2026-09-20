# SIM-MOD-002 Plant

`IPlant` は initialize/reset/step、関節 torque、body linear force、constraint、state のエンジン非依存境界である。`MuJoCoPlant` は一ステップ分の入力を蓄積し、適用後に消去する。`mjModel` と `mjData` へ直接触れるコードは Simulation 内に限定する。

- Joint torque: `qfrc_applied`
- Body force: `mj_applyFT`
- Equality: `eq_active`; equality がなければ有限 PD torque
- Test: `UT-SIM-MOD-002_mujoco_plant.cpp`, `IT-SIM_single_joint.cpp`
