# Muscle-Like Multi-Motor Actuator 共通詳細設計
# 1. 目的
小型BLDC Motorを複数連動し、人体筋肉の起始・停止方向を参考に関節へ力を作用させる共通Actuator構造を定義する。
# 2. Standard Motor Family
| Class | Model | Diameter | Max continuous mechanical power @25°C | Mass |
| :- | :- | --: | --: | --: |
| S | Portescap 22ECT35 | 22 mm | 34 W | 67 g |
| M | Portescap 22ECT48 | 22 mm | 54 W | 98 g |
| L | Portescap 22ECT60 | 22 mm | 86 W | 123 g |
Main Actuator Busは24 V nominalを基本とする。
# 3. Mechanical Principle
```text
Motor(s)
  ↓
Reduction / Screw
  ↓
Linear displacement or Tendon tension
  ↓
Origin bracket ─ Actuator line-of-action ─ Insertion bracket
  ↓
Joint moment
```
基本式:
```text
P = F * v
tau = F * r_eff
```
# 4. Multi-Motor Bundle
同一筋肉Groupに複数Motorを使用する場合、機械的・電気的にLoad Sharingを行う。
- Current balancing
- Position / velocity synchronization
- Force sharing
- Failed-motor isolation
- Thermal derating per motor
# 5. Transmission候補
- Ball Screw
- Lead Screw
- Cable / Tendon
- Pulley
- Linkage
- Differential mechanism
各筋肉Groupごとに必要Force / Velocity / Stroke / Backdrivabilityから選定する。
# 6. Anatomical Mapping Rule
人体筋肉の形状そのものを再現することではなく、以下を機械設計へ写像する。
- approximate origin region
- approximate insertion region
- line of action
- moment arm variation
- agonist / antagonist relation
- multi-joint action where relevant
# 7. Control Interface
Actuator Unitは上位へ以下の抽象Commandを提供する。
- target force / tension
- target displacement
- target velocity
- enable / disable
低レベルMotor current / commutationはDriver側で処理する。
# 8. Safety
- per-motor current limit
- group force/tension limit
- travel limit
- thermal derating
- asymmetric load detection
- tendon slack / break detection
- command timeout
# 9. Power Budget
Installed motor ratingの単純合算をSystemの許容同時出力とはみなさない。
Power ManagerがGroup単位でCurrent / Power allocationを行う。
# 10. TBD
- Screw lead
- Gear ratio
- Tendon material
- Pretension
- Force sensor
- Cooling
- Driver model
