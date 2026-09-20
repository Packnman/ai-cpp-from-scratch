# 下肢 筋肉模倣Hybrid Actuation 詳細設計
# 1. 構成
```text
Hip / Thigh : Muscle-like multi-motor actuators
Knee        : Passive joint + Optional Electromagnetic Lock
Ankle       : Passive Spring-Damper Joint
```
# 2. 設計目的
- Distal motor mass削減
- 消費電力削減
- 人体筋骨格に近い複数Line-of-Actionによる荷重分散
- 倒立振子 / Foot Placement主体の歩行
- Spring / Lockを利用した保持電力低減
# 3. Hip / Thigh Actuator Groups
片脚の初期構成:
| 模倣筋 | 主な機械的役割 | Motor | 本数 |
| :- | :- | :- | --: |
| Gluteus maximus | Posterior hip extension側 | 22ECT60 | 3 |
| Rectus femoris | Anterior thigh / hip-knee連動側 | 22ECT48 | 2 |
| Biceps femoris | Posterior-lateral thigh側 | 22ECT48 | 2 |
| Adductor magnus | Medial thigh / adduction側 | 22ECT48 | 2 |
各Actuatorは人体の起始・停止方向を参考にPelvis / Femur / Tibia側Bracketへ接続する。
# 4. Leg Joint Concept
大腿骨、膝蓋骨、脛骨を主要Joint Elementとして扱う。
Patella-like Elementは力伝達・Moment Arm形成に利用する機械要素として設計し、肩甲骨的な中間要素というコンセプトは保持するが、最終構造はMechanical Detailで確定する。
# 5. Knee
Kneeは常時Motor駆動しない。
遊脚時は受動運動を許容し、必要に応じ支持・姿勢保持でElectromagnetic Lockを使用する。
LockはOptionalとし、以下を満たす場合のみEngageする。
- knee angular velocity within limit
- joint angle within lock window
- contact / gait phase consistent
- lock hardware healthy
- no EmergencyStop contradiction
# 6. Ankle
AnkleはActive Motorを基本的に使用しない。
Spring-Damper:
```text
tau = -k * theta - c * theta_dot
```
着地衝撃吸収、振動減衰、エネルギー蓄積・再利用、受動的地面追従を狙う。
# 7. Control上の扱い
Muscle Actuator commandはContinuous入力、Knee LockはDiscrete入力、AnkleはPassive Dynamicsとして扱う。
したがって下肢はContinuous + Discrete + Passiveを含むHybrid Systemである。
歩行計画、Capture Point、MPC / Hybrid MPC、Lock timing最適化はControl System責務とする。
# 8. Actuator System責務
- Muscle actuator force/position/current command execution
- Knee lock command execution
- Knee / ankle / actuator state reporting
- Safety limit / fault detection
- Thermal / current derating
# 9. 未確定
- Exact bracket coordinates / moment arms
- Stroke
- Screw lead / transmission ratio
- Knee lock型式
- Spring constant k
- Damping coefficient c
- Passive range / mechanical stop
