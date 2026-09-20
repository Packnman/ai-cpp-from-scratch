# CMP-MUS-001 Muscle-Like Multi-Motor Actuator Unit 設計仕様書
# 1. Architecture
```text
BLDC Motor(s)
→ Coupling / Gear
→ Screw / Pulley
→ Slider / Tendon
→ Origin-Insertion Brackets
```
# 2. Standard Motor
CMP-MOT-001を使用する。
# 3. Basic Equations
```text
P = F * v
tau_joint = F * r_eff
```
# 4. Multi-Motor
複数Motor時はCurrent / Position / Velocityを同期し、Load Sharingを行う。
# 5. Control Interface
```text
MuscleActuatorCommand
├── mode
├── target_force
├── target_displacement
├── target_velocity
└── limits
```
# 6. State
```text
MuscleActuatorState
├── displacement
├── velocity
├── estimated_force
├── motor_states[]
├── temperature
├── limit_state
└── fault
```
# 7. 詳細設計
- [Mechanical Architecture](./detail/mechanical_architecture.md)
- [Load Sharing](./detail/load_sharing.md)
- [Sensing](./detail/sensing.md)
- [Safety](./detail/safety.md)
