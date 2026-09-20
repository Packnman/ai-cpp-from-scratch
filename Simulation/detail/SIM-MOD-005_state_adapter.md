# SIM-MOD-005 State Adapter

`MuJoCoStateAdapter` は MuJoCo 配列から値型 `RobotPlantState` を構築する。hinge/slide は scalar、ball/free は先頭値と幅情報に従い安全に読み出す。body pose、world velocity、joint effort、sensor data、contact を一つの timestamped snapshot にまとめる。

pelvis が存在すれば base state とし、なければ world 以外の先頭 body を使用する。

- Test: `UT-SIM-MOD-005_state_adapter.cpp`
