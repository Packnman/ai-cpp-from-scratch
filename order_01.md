# MuJoCoを使用したRobot Simulation Systemの実装

現在のRobot Projectへ、MuJoCo C APIを使用した全身Robot Plant Simulation Systemを追加してください。
今回のSimulationはPythonではなくC/C++で実装してください。
MuJoCoのPython APIは使用しないでください。

## 1. 最初に既存Repositoryを調査すること

実装前に必ず現在のRepository全体を確認してください。
特に以下を調査してください。

```text
Control/
Actuator/
Component/
Simulation/     # 既に存在する場合
CMakeLists.txt
src/
include/
test/
```

既存の実際のFolder構成、namespace、命名規則、CMake構成、Test Frameworkを優先してください。
このPromptに記載されたFolder名とRepositoryが異なる場合は、既存構成へ自然に統合してください。
既存Control / Actuator / Componentを大規模にRewriteしないでください。

## 2. Architecture

責務を以下のように分離してください。

```text
Brain
  ↓
Control
  ↓
Actuator
  ↓
Plant Interface
  ↓
Simulation / MuJoCo
```

各Systemの責務:

```text
Control
    Robot Dynamics上で必要な
    Joint Torque / Position / Velocity / Body Targetを決定
Actuator
    Control要求を実Componentへ変換
    Muscle Force
    Cylinder Force
    Knee Lock
    Motor / Driver Command
Simulation
    Robot Plantの剛体Dynamicsを計算
    Contact
    Gravity
    Joint Constraint
    Body Motion
    Collision
    Sensor Ground Truth
Component
    Motor / Screw / Tendon / Lock / Spring等の物理Parameter定義
```

重要:
ControlとActuatorはMuJoCoへ直接依存させないでください。
MuJoCo依存はSimulation System内部へ閉じ込めてください。

## 3. Dependency Direction

理想的な依存方向:

```text
Control ───────┐
               ↓
Actuator → IPlant / Plant Types
               ↑
               │ implements
Simulation / MuJoCo
```

禁止:

```text
Control → mujoco.h
Actuator → mujoco.h
Component → mujoco.h
```

MuJoCo Headerをincludeしてよいのは原則Simulation側だけとしてください。

## 4. Simulation Folder Structure

既存Repositoryに自然に合わせつつ、基本構成は以下としてください。

```text
Simulation/
├── README.md
├── requirement/
│   └── simulation_requirement.md
├── spec/
│   └── simulation_system_design_spec.md
├── detail/
│   ├── README.md
│   ├── SIM-MOD-001_simulation_manager.md
│   ├── SIM-MOD-002_mujoco_plant.md
│   ├── SIM-MOD-003_robot_model.md
│   ├── SIM-MOD-004_actuator_adapter.md
│   ├── SIM-MOD-005_state_adapter.md
│   ├── SIM-MOD-006_contact_manager.md
│   ├── SIM-MOD-007_sensor_simulator.md
│   ├── SIM-MOD-008_viewer.md
│   └── SIM-MOD-009_log_trace.md
├── include/
│   └── simulation/
│       ├── IPlant.hpp
│       ├── PlantTypes.hpp
│       ├── SimulationManager.hpp
│       ├── MuJoCoPlant.hpp
│       ├── MuJoCoModel.hpp
│       ├── MuJoCoActuatorAdapter.hpp
│       ├── MuJoCoStateAdapter.hpp
│       └── MuJoCoViewer.hpp
├── src/
│   ├── SimulationManager.cpp
│   ├── MuJoCoPlant.cpp
│   ├── MuJoCoModel.cpp
│   ├── MuJoCoActuatorAdapter.cpp
│   ├── MuJoCoStateAdapter.cpp
│   └── MuJoCoViewer.cpp
├── model/
│   ├── humanoid/
│   │   ├── robot.xml
│   │   ├── body.xml
│   │   ├── actuators.xml
│   │   ├── sensors.xml
│   │   ├── contact.xml
│   │   └── assets/
│   │       ├── mesh/
│   │       └── texture/
│   └── test/
│       ├── single_joint.xml
│       ├── double_pendulum.xml
│       └── simple_leg.xml
├── config/
│   ├── simulation.json
│   └── mujoco.json
└── test/
    ├── unit/
    │   ├── UT-SIM-MOD-001_simulation_manager.cpp
    │   ├── UT-SIM-MOD-002_mujoco_plant.cpp
    │   ├── UT-SIM-MOD-004_actuator_adapter.cpp
    │   ├── UT-SIM-MOD-005_state_adapter.cpp
    │   └── UT-SIM-MOD-006_contact_manager.cpp
    └── integration/
        ├── IT-SIM_single_joint.cpp
        ├── IT-SIM_actuator_plant.cpp
        └── IT-SIM_control_actuator_plant.cpp
```

既存ProjectでSource/Headerを別の場所へ集約している場合は、その規則を優先してください。
ただし論理構造は維持してください。

## 5. Core Plant Interface

MuJoCo固有型を外部へ漏らさないPlant Interfaceを定義してください。
概念:

```cpp
class IPlant
{
public:
    virtual ~IPlant() = default;
    virtual void initialize() = 0;
    virtual void reset() = 0;
    virtual void step(double dt) = 0;
    virtual void applyJointTorque(
        JointId joint,
        double torque
    ) = 0;
    virtual void applyLinearForce(
        BodyId body,
        const Vec3& point,
        const Vec3& force
    ) = 0;
    virtual void setConstraintState(
        ConstraintId id,
        ConstraintState state
    ) = 0;
    virtual RobotPlantState getState() const = 0;
};
```

既存Projectに共通Math Type / ID Typeがある場合はそれを再利用してください。

## 6. Plant State

最低限以下を取得可能にしてください。

```text
RobotPlantState
├── simulationTime
├── base
│   ├── position
│   ├── orientation
│   ├── linearVelocity
│   └── angularVelocity
├── joints[]
│   ├── position
│   ├── velocity
│   ├── acceleration
│   └── appliedTorque
├── bodies[]
│   ├── position
│   ├── orientation
│   ├── linearVelocity
│   └── angularVelocity
├── contacts[]
└── sensors[]
```

MuJoCo内部では主に:

```text
mjData::qpos
mjData::qvel
mjData::qacc
```

等から取得してください。
qpos indexとJoint IDを固定値として各所へ散在させないでください。
Modelロード時にMapping Tableを生成してください。

## 7. MuJoCo Wrapper

以下のようなRAII Wrapperを作ってください。

```cpp
class MuJoCoModel
{
private:
    mjModel* _model{};
    mjData* _data{};
public:
    MuJoCoModel(...);
    ~MuJoCoModel();
    MuJoCoModel(const MuJoCoModel&) = delete;
    MuJoCoModel& operator=(const MuJoCoModel&) = delete;
};
```

`mjModel` / `mjData`を直接`new/delete`しないでください。
MuJoCo公式API:

```text
mj_makeData()
mj_deleteData()
mj_deleteModel()
```

等を使用してください。

## 8. MuJoCo Simulation Step

基本実装はMuJoCo C APIを使用してください。
通常:

```cpp
mj_step(model, data);
```

ただしControl / Actuator Outputを現在Stateに基づいて同一Stepへ入力する必要がある場合は:

```cpp
mj_step1(model, data);
// read position / velocity dependent quantities
// Control / Actuator output
// set qfrc_applied / xfrc_applied / ctrl
mj_step2(model, data);
```

を使用可能なArchitectureにしてください。
注意:
`mj_step1()` / `mj_step2()`方式ではIntegrator制約があるため、Integrator選択との整合を確認してください。
初期実装ではEulerまたはImplicit系Integratorを優先してください。

## 9. Simulation Time

SimulationはWall Clockに依存させないでください。

```text
simulation dt
physics dt
control dt
viewer refresh rate
```

を分離してください。
例:

```text
Physics      1 kHz
Actuator     1 kHz
Control      100 Hz
Viewer       60 Hz
```

これは例であり、現在のProject仕様に既定値があればそちらを使ってください。
Simulationの正しさをFPSへ依存させないでください。

## 10. Robot Model

MuJoCo MJCFを使用してRobot Plantを表現してください。
Prototypeでは段階的に構築してください。

```text
Stage 1
single revolute joint
Stage 2
2-link arm
Stage 3
single leg
Stage 4
floating-base simplified humanoid
Stage 5
full humanoid
```

いきなりFull Humanoidだけを作らないでください。

## 11. Full Humanoid Plant

最終的なHumanoid Modelでは最低限以下を持たせる構造にしてください。

```text
World
└── Pelvis / Floating Base
    ├── Torso
    │   ├── Head / Neck
    │   ├── Left Shoulder / Arm
    │   └── Right Shoulder / Arm
    ├── Left Leg
    └── Right Leg
```

各Rigid Bodyについて:

* mass
* center of mass
* inertia tensor
* joint
* joint limit
* collision geometry
* visual geometry
  を設定可能にしてください。
  現時点で未確定な質量・慣性・寸法を根拠なく確定しないでください。
  Config / MJCF ParameterとしてTBDを明示してください。

## 12. Visual MeshとCollision Geometry

実CAD MeshをVisualとして利用できる構成にしてください。
ただしCollisionは可能な限り簡略化してください。
例:

```text
Visual
    real robot mesh
Collision
    capsule
    box
    cylinder
    simplified convex mesh
```

CAD Meshをそのまま複雑なCollision Meshとして使用しないでください。

## 13. Actuator → MuJoCo Interface

Actuator SystemはComponent特性を計算し、Plantへ最終的なPhysical Effortを渡すものとします。
Simulation側は以下を受け取れるようにしてください。

### Rotary Joint

```text
joint torque [Nm]
```

MuJoCoでは必要に応じ:

```cpp
data->qfrc_applied[dofIndex] += torque;
```

を使用してください。

### Force at Attachment Point

Muscle / TendonをGeometryベースで検証する場合:

```text
body
application point
force vector
```

を受け取れるようにしてください。
MuJoCoのExternal Force Interfaceへ変換してください。

### Linear Cylinder

```text
force
attachment point A
attachment point B
```

としてPlantへ適用可能にしてください。

### Knee Lock

Lockは無限Torqueとして実装しないでください。
MuJoCo Constraint / Equality / Joint constraintまたは有限の高Stiffness Modelのどちらを使用するか調査し、Architecture上切替可能にしてください。

### Passive Ankle

Spring-Damper reactionをMuJoCo側またはComponent Model側のどちらで計算するか1箇所へ統一してください。
二重計算を禁止します。

## 14. Component Simulationとの境界

既存Component Document:

```text
CMP-MOT-001
CMP-MUS-001
CMP-DRV-001
CMP-TRN-001
CMP-LIN-001
CMP-LCK-001
CMP-SPR-001
```

をParameter Sourceとして使用してください。
ただし責務は分けます。

```text
Actuator / Component Model
    Motor electrical dynamics
    Screw / transmission
    Tendon force
    Current / thermal limit
    Lock command logic
Simulation / MuJoCo
    Rigid body dynamics
    gravity
    contact
    collision
    joint constraint
    body motion
```

同じDynamicsをActuator側とMuJoCo側の両方で二重に計算しないでください。

## 15. Initial Integration Example

最初のIntegration Targetは1自由度Elbowにしてください。

```text
Control
  ↓ elbow torque request
Actuator
  ↓ Biceps muscle model
  ↓ tendon force / elbow torque
MuJoCoPlant
  ↓ rigid body dynamics
Elbow Joint
  ↓
qpos / qvel
  ↓
Feedback
  ↓
Control
```

モデル:

```text
Upper Arm
Elbow Revolute Joint
Forearm
Gravity
Biceps effective torque input
```

以下を確認してください。

* Torqueを加えるとElbowが動く
* GravityによりForearmが落下する
* qposから角度を取得できる
* qvelから角速度を取得できる
* Torque符号が正しい
* Control / Actuator / MuJoCo間でUnitが一貫する

## 16. MuJoCo Model Mapping

名前でMappingできるRepositoryを作ってください。
例:

```cpp
struct MuJoCoJointBinding
{
    JointId logicalId;
    int jointId;
    int qposAddress;
    int dofAddress;
};
```

同様に:

```text
BodyBinding
SiteBinding
SensorBinding
ActuatorBinding
```

を持てるようにしてください。
Model内部IndexをControl / Actuatorへ公開しないでください。

## 17. Contact

Contact Managerを実装してください。
最低限:

```text
contact body pair
contact position
contact normal
contact force
```

をRobot Stateへ変換可能にしてください。
足裏Contactを将来ControlのFeedbackへ渡せるようにしてください。

## 18. Sensor Simulation

初期段階ではGround Truth Sensorで構いません。
最低限:

* IMU orientation
* angular velocity
* joint position
* joint velocity
* foot contact
  を取得可能な設計にしてください。
  将来:

```text
noise
bias
delay
dropout
```

を追加できる構造にしてください。
Sensor SimulationはControl SystemへMuJoCo固有型を渡してはいけません。

## 19. Viewer / Visualization

MuJoCoの公式Visualization APIを利用したViewerを追加してください。
最低限:

* Robot表示
* Camera rotate / zoom
* pause / resume
* single step
* simulation reset
  を可能にしてください。
  可能ならDebug表示:
* joint axes
* contact point
* contact force
* center of mass
* tendon/site
  も切り替え可能にしてください。
  ViewerのFPSとPhysics Simulation周期を分離してください。
  Headless実行も可能にしてください。

```text
--headless
--viewer
```

等の切替をConfigまたはCLIで行えるようにしてください。

## 20. Future Backend

Simulation SystemをMuJoCo固定Architectureにしないでください。

```text
IPlant
├── MuJoCoPlant
├── SimplePlant
├── Future UnrealPlant
└── Future RealRobotPlant Adapter
```

のように将来交換可能にしてください。
ただし今回はMuJoCoPlantのみ本実装してください。

## 21. Real Robotとの関係

Real RobotはPhysics Stepを計算しないため、厳密にはMuJoCoPlantと同一実装にはなりません。
それでもControl / Actuator上位側から見える

```text
apply effort
read state
```

のBoundaryを共通化できるように設計してください。
必要なら:

```text
IPlant
ISimulationPlant : IPlant
```

のようにSimulation特有`step()`を分離して構いません。
既存Architectureに最も自然なものを選択してください。

## 22. CMake / Dependency

MuJoCoをC/C++ dependencyとして追加してください。
OS / Dev Container / Existing Build Systemを調査し、現在のProjectに最も自然な方法を選んでください。
実装後:

```text
configure
build
unit test
integration test
```

が再現可能であることを確認してください。
MuJoCoのVersionをBuild / Documentation上で明示してください。

## 23. Test

### Unit

最低限:

```text
MuJoCo model load
joint mapping
body mapping
state conversion
torque application
reset
time step
contact conversion
```

### Integration

#### IT-SIM-001 Single Joint

Torque → MuJoCo → qpos/qvel。

#### IT-SIM-002 Actuator Plant

Actuator output → MuJoCo elbow → Feedback。

#### IT-SIM-003 Gravity

No torqueでGravity response確認。

#### IT-SIM-004 Contact

Simple leg / footがfloor contactすること。

#### IT-SIM-005 Determinism

同一初期状態・同一入力で同一結果になること。

#### IT-SIM-006 Headless

ViewerなしでもSimulationが実行できること。

## 24. Numerical Validation

単純Modelについて解析解または既知結果と比較してください。
例:

```text
single pendulum
free fall
constant torque rigid body
```

MuJoCoを呼べたことだけをTest成功条件にしないでください。
Physics結果が合理的であることを検証してください。

## 25. Logging

Simulation Log:

```text
simulation time
qpos
qvel
qacc
applied effort
contacts
control cycle
physics cycle
numerical warning
```

高頻度Logを常時Consoleへ大量出力しないでください。
必要に応じCSV / Binary / Project既存Logへ出力してください。

## 26. Unit

全SystemでSI単位を使用してください。

```text
length       m
mass         kg
time         s
force        N
torque       Nm
velocity     m/s
angular vel  rad/s
angle        rad
```

MJCF / Control / Actuator間のUnit変換を暗黙に行わないでください。

## 27. Documentation

以下を作成・更新してください。

```text
Simulation/README.md
Simulation/requirement/simulation_requirement.md
Simulation/spec/simulation_system_design_spec.md
Simulation/detail/README.md
Simulation/detail/SIM-MOD-*.md
```

`detail/README.md`をSimulation詳細設計の親文書とし、各Module文書へリンクしてください。
READMEには最低限:

* Architecture
* Module一覧
* Dependency
* Simulation Loop
* Folder Structure
* Build
* Run
* Viewer
* Headless
* Test
  を記載してください。

## 28. Simulation Modules

以下をBaselineとしてください。

```text
SIM-MOD-001 Simulation Manager
SIM-MOD-002 MuJoCo Plant
SIM-MOD-003 Robot Model Manager
SIM-MOD-004 Actuator Adapter
SIM-MOD-005 State Adapter
SIM-MOD-006 Contact Manager
SIM-MOD-007 Sensor Simulator
SIM-MOD-008 Viewer
SIM-MOD-009 Log / Trace
```

責務が既存Repositoryと合わない場合は調整可能ですが、変更理由を最終報告してください。

## 29. Implementation Phases

この順で進めてください。

```text
Phase 1
Repository / Control / Actuator Interface調査
Phase 2
MuJoCo Dependency + Minimal Model Load
Phase 3
IPlant + MuJoCoPlant
Phase 4
Single Joint MJCF
Phase 5
Torque → qpos/qvel Integration Test
Phase 6
Actuator Adapter接続
Phase 7
State Adapter / Contact
Phase 8
Viewer
Phase 9
Simple Leg
Phase 10
Simplified Humanoid Skeleton
Phase 11
Documentation
```

各PhaseでBuild / Testしてください。

## 30. 今回やらないこと

以下は今回のScope外です。

* 完成版Full CAD Humanoid
* Camera Image Recognition
* VLA / Brain
* Reinforcement Learning
* Unreal Engine integration
* 高精度Sensor Noise Model
* 完成版Walking Controller
* 実機通信
  ただし将来追加できるInterfaceは維持してください。

## 31. 禁止事項

* ControlへMuJoCo型を漏らす
* ActuatorへMuJoCo型を漏らす
* MuJoCo IndexをSystem全体へ直書きする
* PhysicsとViewer更新周期を同一に固定する
* Motor DynamicsをMuJoCoとActuator双方で二重計算する
* 不明な質量・慣性を実機確定値として捏造する
* TestをViewer目視だけで済ませる
* Simulation専用if文をControl / Actuatorへ大量追加する
* Python implementationへ変更する

## 32. 最終報告

完了時に以下を報告してください。

1. 調査した既存Architecture
2. MuJoCo Version
3. 新規Folder / File
4. 変更File
5. Simulation Module構成
6. IPlant Interface
7. Control / Actuatorとの接続方法
8. MuJoCo Model構成
9. Viewer実行方法
10. Headless実行方法
11. Unit Test結果
12. Integration Test結果
13. 未確定Robot Parameter
14. Full Humanoid化に必要な次の作業

# 最終目標

最終Architectureは以下としてください。

```text
Brain
  ↓
Control
  ↓
Actuator
  ↓
Physical Effort
  ↓
IPlant
  ↓
MuJoCoPlant
  ↓
MuJoCo Rigid Body Dynamics
  ↓
RobotPlantState
  ↓
Feedback
  └────────→ Control
```

MuJoCoはRobot Plantの剛体Dynamics / Contact / Constraintを担当し、ControlとActuatorは実機とSimulationで可能な限り同じCode Pathを使用してください。
