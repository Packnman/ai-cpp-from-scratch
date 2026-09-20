# Feedback / State Estimation
# 1. Sources
- Sensor System
- Actuator System
- Power System
# 2. ControlState
```text
ControlState
├── body_attitude
├── body_angular_velocity
├── joint_states[]
├── muscle_actuator_states[]
├── cylinder_states[]
├── knee_lock_states[]
├── contact_states[]
├── ankle_spring_states[]
├── power_limit_state
└── timestamp
```
# 3. Estimation
State Estimatorは将来EKF等を追加可能とする。
Prototypeでは測定値統合 + FilterをBaselineとする。
