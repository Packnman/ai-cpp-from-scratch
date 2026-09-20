# CMP-SPR-001 Ankle Passive Spring-Damper Component 要求仕様書
# 1. 目的
AnkleへActive Motorを置かず、Spring / DamperでCompliance、衝撃吸収、Energy Returnを実現する。
# 2. Functional
- restoring torque
- damping torque
- mechanical travel limit
# 3. Model
```text
tau = -k * theta - c * theta_dot
```
# 4. Performance
k / c / travelは30 kg Robotの歩行・着地要求から決定する。
# 5. Safety
Bottom-out / Over-travelをMechanical Stopで防止する。
