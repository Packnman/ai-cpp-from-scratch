# Component System 文書セット
# 1. 目的
Actuator Systemから物理Component実装詳細を分離し、Component単位で要求仕様・設計仕様・詳細設計を管理する。
# 2. 文書責務
```text
Actuator System
  → Componentをどう使うか / どう協調させるか
Component Documents
  → Component自体が何を満たすか / どう構成するか / どう実装するか
```
# 3. Component一覧
| ID | Component | 概要 |
| :- | :- | :- |
| CMP-MOT-001 | [CMP-MOT-001 Standard BLDC Motor Component](./CMP-MOT-001_motor/README.md) | Portescap 22ECT35 / 22ECT48 / 22ECT60を標準Motor Familyとして定義する。 |
| CMP-MUS-001 | [CMP-MUS-001 Muscle-Like Multi-Motor Actuator Unit](./CMP-MUS-001_muscle_actuator_unit/README.md) | 複数BLDC、Transmission、Sensorを一体化した筋肉模倣Actuator Unit。 |
| CMP-DRV-001 | [CMP-DRV-001 BLDC Motor Driver Component](./CMP-DRV-001_motor_driver/README.md) | 22ECT Familyを駆動する24 V Current-Controlled BLDC Driver。 |
| CMP-TRN-001 | [CMP-TRN-001 Muscle Transmission Component](./CMP-TRN-001_transmission/README.md) | Motor回転をLinear Force/Tendon Tensionへ変換するTransmission。 |
| CMP-LIN-001 | [CMP-LIN-001 Waist Electric Linear Cylinder Component](./CMP-LIN-001_waist_linear_cylinder/README.md) | 腰Pitch/Rollを生成する左右2本の電動Linear Cylinder。 |
| CMP-LCK-001 | [CMP-LCK-001 Knee Electromagnetic Lock Component](./CMP-LCK-001_knee_lock/README.md) | Passive Knee Jointを必要時に固定するOptional Electromagnetic Lock。 |
| CMP-SPR-001 | [CMP-SPR-001 Ankle Passive Spring-Damper Component](./CMP-SPR-001_ankle_spring_damper/README.md) | 足首のPassive Complianceと着地衝撃吸収を担うComponent。 |
# 4. 基本階層
```text
Component/
├── Component-A/
│   ├── requirement_spec.md
│   ├── design_spec.md
│   └── detail/
│       ├── README.md
│       └── *.md
└── Component-B/ ...
```
# 5. Traceability
各Component RequirementはComponent Designへ、Component DesignはDetail DesignへTrace可能な構成とする。
# 6. Actuator Systemとの境界
- Actuator SystemはComponent IDとInterfaceを参照する。
- Motor型番、Transmission構造、Driver内部、Lock機構、Spring構造はComponent文書を正とする。
- Actuator System側にはMuscle Group配置、Command、State、Safety連携のみを残す。
