# Actuator System Simulation Component実装

現在のリポジトリに、Actuator Systemから利用できるSimulation用Component Model / Driver Backendを実装してください。

## 1. 最初に行うこと

実装前に必ず既存リポジトリを調査してください。
特に以下を確認してください。

```text
Actuator/
├── requirement/
├── spec/
├── detail/
│   ├── ACT-MOD-001_command_receiver.md
│   ├── ACT-MOD-002_command_validator.md
│   ├── ACT-MOD-003_actuator_manager.md
│   ├── ACT-MOD-004_drive_interface.md
│   ├── ACT-MOD-005_state_monitor.md
│   ├── ACT-MOD-006_safety_limiter.md
│   ├── ACT-MOD-007_fault_monitor.md
│   ├── ACT-MOD-008_power_interface.md
│   ├── ACT-MOD-009_log_trace.md
│   └── README.md
└── component/
    ├── CMP-MOT-001_motor/
    ├── CMP-MUS-001_muscle_actuator_unit/
    ├── CMP-DRV-001_motor_driver/
    ├── CMP-TRN-001_transmission/
    ├── CMP-LIN-001_waist_linear_cylinder/
    ├── CMP-LCK-001_knee_lock/
    └── CMP-SPR-001_ankle_spring_damper/
```

既存Source/Header/Testの実際のディレクトリ構成・namespace・命名規則・Interfaceを調査し、それに合わせて実装してください。
文書に書かれている想定パスより、実際のリポジトリ構成を優先してください。
大規模な既存コードの移動・Rename・再設計は行わないでください。

## 2. 設計上の責務

今回の責務分離は以下です。

```text
Control System
    ↓
Joint Torque / Position / Velocity等のPlant要求
    ↓
Actuator System
    ↓
実際のComponent特性を使ってHardware Commandへ変換
    ↓
Component
    ↓
Motor / Transmission / Tendon / Lock / Spring
    ↓
Plant
```

Simulationでもこの境界を維持してください。
Control Systemの責務をActuator Simulationへ持ち込まないでください。
今回作るSimulation Componentは「ロボット全身Dynamics Simulator」ではありません。
Actuator Systemから見たPhysical Componentを数式で模擬するBackendです。
つまり以下を行います。

```text
Actuator Command
→ Virtual Motor
→ Virtual Transmission
→ Virtual Tendon / Linear Actuator
→ Force / Torque / Position / Velocity State
```

Robot Body全体の剛体運動、接触、重力、歩行DynamicsそのものはPlant Simulator側の責務とし、本実装ではInterfaceまでにしてください。

## 3. 既存Actuator Interfaceとの統合

現在の詳細設計では `ACT-MOD-003 Actuator Manager` が概念的に以下を保持しています。

```cpp
struct ActuatorEntry
{
    ActuatorDescriptor descriptor;
    ActuatorState state;
    std::unique_ptr<IActuatorDriver> driver;
};
```

また `ACT-MOD-004 Drive Interface` は概念的に以下です。

```cpp
class IActuatorDriver
{
public:
    virtual ~IActuatorDriver() = default;
    virtual void configure(const ActuatorDescriptor&) = 0;
    virtual void enable() = 0;
    virtual void disable() = 0;
    virtual void command(const DriveCommand&) = 0;
    virtual void stop(StopMode mode) = 0;
    virtual ActuatorState readState() = 0;
    virtual void resetFault() = 0;
};
```

既存コードに実際のInterfaceが存在する場合は、それを変更せず最大限再利用してください。
Simulation BackendはPhysical Driverと交換可能にしてください。
目標:

```text
IActuatorDriver
├── MockActuatorDriver
├── SimActuatorDriver
└── PhysicalActuatorDriver
```

ただしActuation Typeが異なるため、必要なら既存設計に沿って以下へ分けても構いません。

```text
ISimActuatorDriver
├── SimMuscleActuatorDriver
├── SimLinearCylinderDriver
├── SimKneeLockDriver
└── SimPassiveElasticAdapter
```

既存Interfaceとの整合を最優先してください。

## 4. 実装するSimulation Component

最低限、以下を実装してください。

```text
SimBLDCMotor
SimTransmission
SimMuscleActuator
SimWaistLinearCylinder
SimKneeLock
SimSpringDamper
```

必要なら以下も作成してください。

```text
SimMotorDriver
SimTendon
SimComponentFactory
SimComponentRegistry
SimComponentConfig
```

## 5. SimBLDCMotor

Component資料:

```text
Actuator/component/CMP-MOT-001_motor/
```

に記載された値を使用してください。
Prototype 1 Baseline:

```text
S : Portescap 22ECT35 10B 80 01
M : Portescap 22ECT48 10B 35 01
L : Portescap 22ECT60 10B 21 01
```

最低限以下のMotor Parameterを持たせてください。

```cpp
struct MotorParameters
{
    double resistance;
    double inductance;
    double torqueConstant;
    double backEmfConstant;
    double rotorInertia;
    double viscousFriction;
    double nominalVoltage;
    double continuousCurrent;
    double maxSpeed;
    double maxWindingTemperature;
};
```

資料上にviscousFriction等が存在しない場合は勝手に確定値を作らず、Configurable Parameterまたは明示的なDefault Model Parameterとしてください。
Motor Dynamicsは最低限以下を表現してください。

```text
V = R i + L di/dt + Ke ω
τm = Kt i
J dω/dt = τm - τload - Bω
dθ/dt = ω
```

数値積分は初期実装ではSemi-Implicit EulerまたはRK4のどちらかを選択してください。
既存ProjectにIntegratorがある場合は再利用してください。
`dt`を外部から受け取る決定論的実装にしてください。
Wall Clockに依存させないでください。
Current Limit、Voltage Limit、Speed Limitを適用してください。

## 6. SimTransmission

資料:

```text
Actuator/component/CMP-TRN-001_transmission/
```

を参照してください。
初期実装ではScrew Driveを優先してください。
最低限:

```cpp
struct ScrewTransmissionParameters
{
    double lead;
    double efficiency;
    double gearRatio;
    double backlash;
    double minPosition;
    double maxPosition;
};
```

基本変換:

```text
motor rotation
→ gear
→ screw rotation
→ linear displacement
```

理想モデルでは概念的に

```text
x = theta_screw * lead / (2π)
v = omega_screw * lead / (2π)
F ≈ 2π * efficiency * tau_screw / lead
```

を使用できます。
ただし符号・Gear Ratio定義は1箇所で明確にしてください。
Mechanical Stopを実装してください。

## 7. SimMuscleActuator

資料:

```text
CMP-MUS-001_muscle_actuator_unit
CMP-MOT-001_motor
CMP-TRN-001_transmission
```

をCompositionしてください。

```text
SimMuscleActuator
├── 1..N SimBLDCMotor
├── SimTransmission
├── Tendon / Slider State
└── Sensor State
```

Multi-Motor Actuatorに対応してください。
初期実装ではMotorが同一Shaft / Transmissionを駆動する理想Load Sharing Modelで構いません。
ただし後でMotor個別差を追加できる構造にしてください。
State例:

```cpp
struct MuscleActuatorState
{
    double displacement;
    double velocity;
    double tendonForce;
    double estimatedJointTorque;
    std::vector<MotorState> motors;
    bool lowerLimit;
    bool upperLimit;
    bool fault;
};
```

Actuator System側からはMotor個別CommandではなくMuscle Actuatorとして扱えるようにしてください。

## 8. Joint TorqueとMuscle Forceの変換境界

Control SystemからActuator SystemへJoint Torque要求が入る場合、
Actuator System側で

```text
τjoint
→ required tendon force
→ transmission torque
→ motor torque
→ motor current
```

へ変換できる構造にしてください。
基本関係:

```text
τjoint = Fmuscle * r_eff
```

ただしMoment Arm `r_eff` はJoint Angleに依存できるようInterface化してください。
初期実装ではConstant Moment Arm Modelで構いません。
例:

```cpp
class IMomentArmModel
{
public:
    virtual ~IMomentArmModel() = default;
    virtual double momentArm(double jointPosition) const = 0;
};
```

将来、Geometry Modelへ交換可能にしてください。

## 9. SimWaistLinearCylinder

資料:

```text
CMP-LIN-001_waist_linear_cylinder
```

を参照してください。
22ECT60 + Screw Transmissionを使用するCompositionとして実装してください。
最低限:

* length
* velocity
* estimated force
* current
* temperature
* min/max stroke
* mechanical stop
  を表現してください。
  左右2本の協調制御そのものはControl/Actuator上位層の責務なので、Component Modelは1本単位で実装してください。

## 10. SimKneeLock

資料:

```text
CMP-LCK-001_knee_lock
```

を参照してください。
最低限:

```text
Released
Engaging
Locked
Releasing
Fault
```

を表現してください。
入力:

```text
lock()
release()
```

状態:

```text
lockState
jointPosition
jointVelocity
holdingTorqueLimit
```

初期SimulationではLocked時に無限剛性を使わず、有限の高いConstraint/Stiffnessまたは明示的Constraint Stateを返してください。
Plant SimulatorがConstraintを適用できるInterfaceを用意してください。
高Angular VelocityでLock Commandを受けた場合の扱いはComponent仕様に沿って安全側へしてください。

## 11. SimSpringDamper

資料:

```text
CMP-SPR-001_ankle_spring_damper
```

を参照してください。
最低限:

```text
τ = -k θ - c θdot
```

を実装してください。
Parameter:

```cpp
struct SpringDamperParameters
{
    double springConstant;
    double dampingCoefficient;
    double neutralPosition;
    double minPosition;
    double maxPosition;
};
```

Mechanical Stopも考慮してください。
これはPassive ComponentなのでDrive Commandは持ちません。
State / Reaction Torqueを返すAdapterとして実装してください。

## 12. Component Parameter管理

Componentの数値をSource Code中へ散在させないでください。
以下のどちらか、既存Project方針に合う方法を使用してください。

```text
C++ configuration struct
JSON
YAML
SQLite
既存Config system
```

少なくとも以下を識別可能にしてください。

```text
CMP-MOT-001
CMP-MUS-001
CMP-TRN-001
CMP-LIN-001
CMP-LCK-001
CMP-SPR-001
```

Motorについては

```text
22ECT35-80
22ECT48-35
22ECT60-21
```

をParameter Presetとして作ってください。
Component資料の値とSource CodeのParameter値がTrace可能になるようにしてください。

## 13. Simulation / Physical切替

Actuator System本体へ `if (simulation)` を大量に入れないでください。
Dependency Injection / Factoryで切り替えてください。
理想:

```cpp
std::unique_ptr<IActuatorDriver> createActuatorDriver(
    const ActuatorDescriptor& descriptor,
    BackendType backend
);
```

```cpp
enum class BackendType
{
    Simulation,
    Physical
};
```

Simulation / Physicalの違いはDriver / Component Backend以下で吸収してください。
Actuator Manager、Safety Limiter、Fault Monitor等の上位処理は共通化してください。

## 14. Plant Interface

Component SimulationとRobot Plant Simulationを分離してください。
Actuator側からPlantへ最低限以下を渡せるInterfaceを設けてください。

```text
joint torque
linear force
constraint / lock state
passive reaction torque
```

Plant側からActuator Simulationへは

```text
joint position
joint velocity
external load torque
contact/load state if required
```

を戻せるようにしてください。
ただし今回Robot全身Plant Dynamicsそのものは実装対象外です。
テスト用のFakePlant / SimpleSingleJointPlantは作って構いません。

## 15. Safety / Fault

既存:

```text
ACT-MOD-006 Safety Limiter
ACT-MOD-007 Fault Monitor
```

を再利用してください。
Simulation Componentでも以下を再現してください。

* over-current
* over-speed
* over-travel
* over-temperature
* stalled motor
* tendon overload
* lock failure
* sensor invalid
  Fault Injection可能な構造にしてください。
  例:

```cpp
SimFaultInjection
{
    bool hallFailure;
    bool currentSensorFailure;
    bool tendonBreak;
    bool lockFailure;
};
```

Test専用APIでも構いません。

## 16. Thermal Model

初期実装では高精度熱解析は不要ですが、Motor CurrentからCopper Lossを求める一次遅れThermal Modelを実装できる構造にしてください。
概念:

```text
Pcu = I²R
Cth dT/dt = Pcu - (T - Tambient)/Rth
```

Component資料にあるThermal Resistance / Time Constantを利用できるようにしてください。
不明なParameterは勝手な物理定数として固定せずConfigurableにしてください。

## 17. Tests

既存Test Frameworkを使用してください。
最低限以下のUnit Testを追加してください。

### SimBLDCMotor

* zero voltage
* positive voltage acceleration
* back EMF
* current limit
* speed limit
* load torque
* deterministic dt

### SimTransmission

* rotation → displacement
* motor torque → linear force
* gear ratio
* mechanical stop

### SimMuscleActuator

* single motor
* multi motor
* force generation
* stroke limit
* tendon break fault

### SimWaistLinearCylinder

* extension/retraction
* force
* stroke limit

### SimKneeLock

* lock
* release
* invalid engage condition
* fault

### SimSpringDamper

* neutral position
* spring torque
* damping torque
* mechanical limit

## 18. Integration Test

最低1つ、以下のPipeline Testを追加してください。

```text
DriveCommand
→ Actuator System
→ SimActuatorDriver
→ SimMuscleActuator
→ SimBLDCMotor
→ SimTransmission
→ FakeSingleJointPlant
→ ActuatorState
```

例として肘1自由度を使用してください。

```text
Biceps muscle actuator
22ECT35 ×2
Screw transmission
Constant moment arm
Elbow single joint plant
```

Torque Commandを与え、

* Motor Currentが生成される
* Tendon Forceが生成される
* Joint TorqueがPlantへ渡る
* Joint Position / Velocityが変化する
* Actuator StateへFeedbackされる
  ことを確認してください。

## 19. Architecture上の禁止事項

以下は行わないでください。

* Control SystemにPortescap型番を直接依存させる
* Actuator ManagerにMotor数式を直接書く
* Motor Parameterを各Controllerへ重複定義する
* Simulation専用処理をActuator System全体へ散在させる
* Physical Driver Interfaceを壊す
* Robot全身DynamicsをActuator Component内に実装する
* 不明なComponent値を根拠なく確定する
* 既存設計を無視した大規模Rewrite

## 20. Documentation Update

実装後、現在の文書構成を維持したまま必要箇所を更新してください。
特に:

```text
Actuator/detail/README.md
Actuator/detail/ACT-MOD-003_actuator_manager.md
Actuator/detail/ACT-MOD-004_drive_interface.md
Actuator/component/*/detail/
```

へSimulation Backendとの関係を追記してください。
実装Sourceが文書と乖離しないようにしてください。

## 21. 最終報告

完了時に以下を報告してください。

1. 調査した既存構成
2. 採用したArchitecture
3. 新規作成File
4. 変更File
5. Component Model一覧
6. Simulation / Physical切替方法
7. 使用したMotor Parameter
8. Test結果
9. 未確定Parameter
10. 次にPlant Simulatorへ接続するために必要なInterface

## 22. 実装優先順位

一度に過度に作り込まず、以下の順で実装してください。

```text
Phase 1
Component Parameter / Common Interface
Phase 2
SimBLDCMotor
Phase 3
SimTransmission
Phase 4
SimMuscleActuator
Phase 5
SimActuatorDriver + Actuator System統合
Phase 6
SimWaistLinearCylinder
Phase 7
SimKneeLock
Phase 8
SimSpringDamper
Phase 9
Unit Tests
Phase 10
Single Joint Integration Test
Phase 11
Documentation
```

各Phase終了時にBuild / Testし、既存Testを壊していないことを確認しながら進めてください。
最終的に、実機Driverへ差し替えてもActuator System上位層を変更しないArchitectureにしてください。
