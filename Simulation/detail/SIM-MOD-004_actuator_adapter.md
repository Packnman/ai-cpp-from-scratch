# SIM-MOD-004 Actuator Adapter

`MuJoCoActuatorAdapter` は Actuator の `PlantOutput` を `IPlant` 操作へ変換する。joint/passive torque は加算、linear force は direction を正規化し body の application point に適用、constraint は target/stiffness/damping/limit を渡す。ゼロ方向ベクトルは適用しない。

この依存方向により Actuator は Simulation と MuJoCo を知らない。

- Test: `UT-SIM-MOD-004_actuator_adapter.cpp`, `IT-SIM_actuator_plant.cpp`
