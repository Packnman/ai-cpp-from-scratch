# 首 筋肉模倣Actuation 詳細設計
# 1. 目的
首はSternocleidomastoidとSplenius capitisを主要Muscle-Like Actuatorとして構成し、軽量な頭部Displayを支持・姿勢制御する。
# 2. 採用筋
| 模倣筋 | 配置方針 | Motor | 本数/側 |
| :- | :- | :- | --: |
| Sternocleidomastoid | Sternum/Clavicle側からMastoid近傍方向 | 22ECT35 | 1 |
| Splenius capitis | Upper thoracic/cervical spine側からMastoid/Occipital方向 | 22ECT35 | 1 |
Platysmaは採用しない。
# 3. Head
顔はMonitor / Display形式を維持する。
Head mass / CGはTBD。
# 4. Control
左右Actuatorの協調・拮抗によりPitch / Yaw / Rollを生成する構成を検討する。
最終DOFはFree Joint / Universal Joint等のMechanical Design確定後に決める。
# 5. State
- head orientation
- head angular velocity
- actuator displacement
- motor current
- actuator temperature
# 6. Safety
- Neck torque / force limit
- Joint range limit
- Cable/Tendon tension limit if used
- Head collision
- EmergencyStop時のSafe Hold / Release sequence
# 7. TBD
- Joint type
- Exact origin/insertion brackets
- Stroke / moment arm
- Head mass / inertia
