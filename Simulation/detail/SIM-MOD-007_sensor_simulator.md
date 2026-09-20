# SIM-MOD-007 Sensor Simulator

`GroundTruthSensorSimulator` は Plant snapshot から joint/body/contact/sensor の ground-truth view を返す。実センサのノイズや遅延を入れず、Control/Sensor subsystem の比較基準にする。将来のノイズモデルはこの値を入力にする別 decorator として追加する。

- Source: `RobotPlantState`
- Clock: Plant simulation time
- Test coverage: state/contact adapter tests
