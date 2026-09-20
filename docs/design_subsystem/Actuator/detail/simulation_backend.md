# Actuator Simulation Backend 詳細設計

## 1. 境界

Simulation BackendはActuator Systemから見える物理Componentを決定論的に模擬する。
全身剛体Dynamics、接触、重力、歩行DynamicsはPlant Simulatorの責務であり、本Backendには含めない。

```text
DriveCommand
  -> ActuatorManager
  -> IActuatorDriver
       +-> SimActuatorDriver
       `-> PhysicalActuatorDriver (factory injection)
  -> SimMuscleActuator
  -> SimBLDCMotor / SimTransmission
  -> PlantOutput
```

Simulation固有の時間発展は `ISimActuatorDriver::advance(dt, feedback)` に閉じ込める。
`IActuatorDriver` はPhysical Driverと共通のconfigure / enable / command / state APIを維持する。

## 2. 時間・積分

- `dt` は呼出側から秒で明示的に渡す。
- Wall Clockを積分に使用しない。
- Motorは電流、角速度、角度の順に更新するSemi-Implicit Eulerを使用する。
- 剛体Plantの積分は `Simulation/MuJoCoPlant` に委譲する。

## 3. Plant境界

`PlantOutput` はjoint torque、linear force、passive reaction torque、有限剛性Lock constraintを渡す。
`PlantFeedback` はjoint position、joint velocity、external load torqueを戻す。
Lockは無限剛性を使わず、stiffnessとholding torque limitをPlantへ通知する。

## 4. Parameter管理

Motor presetはCMP-MOT-001の資料値をSI単位へ変換し、`motorParameters()` に集約する。
Transmission lead / ratio / efficiency / backlash / stroke、Moment Arm、Lock、Spring-Damperは実機選定前のためConfigurationで注入する。
`defaultElbowMuscleConfig()` の数値は製品確定値ではなく、Pipeline Test用の明示的Simulation Defaultである。

## 5. Fault

自然発生するover-speed、over-travel、over-temperature、tendon overloadに加え、Test APIで
over-current、stall、sensor invalid、tendon break、lock failureを注入できる。
上位層へは共通 `ActuatorFaultCode` と `ActuatorStatus` で通知する。

## 6. 実装位置

- 共通型: `include/actuator/common/`
- Manager: `include/actuator/manager/`, `src/actuator/manager/`
- Driver境界とSimulation Driver: `include/actuator/driver/`, `src/actuator/driver/`
- Simulation Component: `include/actuator/component/sim/`, `src/actuator/component/sim/`
- 剛体Plant、接触、重力: `Simulation/`
- Test: `tests/actuator/`

