# CMP-MUS-001 Muscle-Like Multi-Motor Actuator Unit 要求仕様書
# 1. 目的
人体筋肉の起始・停止方向を模倣する線形力生成Componentの要求を定義する。
# 2. 機能要求
| ID | 要求 |
| :- | :- |
| REQ-CMP-MUS-FUN-001 | 1台以上の標準BLDC Motorを連動可能であること |
| REQ-CMP-MUS-FUN-002 | Rotary motionをLinear displacementまたはTendon tensionへ変換できること |
| REQ-CMP-MUS-FUN-003 | Force / Displacement / Velocity commandの少なくとも1方式を実装可能であること |
| REQ-CMP-MUS-FUN-004 | Multiple Motor時にLoad Sharingできること |
# 3. Mechanical要求
- Origin bracket / Insertion bracket間へ設置可能であること。
- Stroke limitを持つこと。
- Mechanical stopを持つこと。
- Backlash / complianceをCharacterize可能であること。
# 4. Sensor要求
- displacement
- motor current
- temperature
- tension/forceはOption
# 5. Safety要求
- over-travel
- over-current
- over-temperature
- tendon slack / break
- asymmetric motor load
# 6. Performance要求
性能値は配置筋肉ごとに必要Force / Velocity / Strokeを設定する。
# 7. 未確定
- Screw vs Tendon architecture
- Force Sensor方式
- Housing material
