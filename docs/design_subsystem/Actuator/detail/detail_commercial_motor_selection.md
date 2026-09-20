# Muscle Actuator / Commercial Motor Component Selection
# 1. 目的
本書は30 kg級Humanoid Robotの筋肉模倣Actuatorに使用する市販Motorの初期採用構成を定義する。
本構成はPrototype 1のBaselineとし、Link寸法、Moment Arm、Ball Screw / Lead Screw、Cable / Tendon Routing、熱設計の確定後に本数および減速比を再調整する。
# 2. 設計目標
| 項目 | 目標 |
| :- | :- |
| Robot mass | 30 kg |
| 通常運用時間 | 5 h以上 |
| 激しい運動 | 45 min以上 |
| Battery usable ratio | 0.8 |
| 片腕保持性能 | 手先3 kg級を目標 |
| Actuator形式 | 小型BLDC複数連動＋直動/腱伝達 |
| Main actuator voltage | 24 V nominal |
| 手・首最終機構 | 一部TBD |
# 3. 標準Motor Family
Prototype 1ではPortescap 22ECT Ultra ECシリーズを標準Motor Familyとする。
| Class | Model | Diameter | Length | Max continuous mechanical power @25°C | Weight | Nominal voltage候補 |
| :- | :- | --: | --: | --: | --: | :- |
| S | 22ECT35 | 22 mm | 35 mm | 34 W | 67 g | 24 V winding |
| M | 22ECT48 | 22 mm | 48 mm | 54 W | 98 g | 24 V winding |
| L | 22ECT60 | 22 mm | 60 mm | 86 W | 123 g | 24 V winding |
同一22 mm径を基本とし、必要出力に応じてMotor長を変更する。
# 4. 初期配置
## 4.1 首
| 模倣筋 | Model | 本数/側 | 全身本数 |
| :- | :- | --: | --: |
| Sternocleidomastoid / 胸鎖乳突筋 | 22ECT35 | 1 | 2 |
| Splenius capitis / 頭板状筋 | 22ECT35 | 1 | 2 |
## 4.2 上肢・肩甲帯
| 模倣筋 | Model | 本数/側 | 全身本数 |
| :- | :- | --: | --: |
| Biceps brachii / 上腕二頭筋 | 22ECT35 | 2 | 4 |
| Deltoid / 三角筋 | 22ECT35 | 3 | 6 |
| Serratus anterior / 前鋸筋 | 22ECT35 | 2 | 4 |
| Trapezius / 僧帽筋 | 22ECT35 | 2 | 4 |
| Pectoralis major / 大胸筋 | 22ECT48 | 2 | 4 |
| Latissimus dorsi / 広背筋 | 22ECT48 | 2 | 4 |
肘伸展側の拮抗Actuatorは現時点TBDとし、Prototypeの可動試験で必要性を評価する。
## 4.3 下肢
| 模倣筋 | Model | 本数/側 | 全身本数 |
| :- | :- | --: | --: |
| Gluteus maximus / 大殿筋 | 22ECT60 | 3 | 6 |
| Rectus femoris / 大腿直筋 | 22ECT48 | 2 | 4 |
| Biceps femoris / 大腿二頭筋 | 22ECT48 | 2 | 4 |
| Adductor magnus / 大内転筋 | 22ECT48 | 2 | 4 |
膝はActive Motor JointではなくOptional Electromagnetic Lockを基本とする。
足首はPassive Spring-Damperを基本とする。
## 4.4 腰
左右2本のLinear CylinderでPitch / Rollを生成する。
Prototype 1の駆動Motor候補は各Cylinderにつき22ECT60 ×1とする。
最終採否は必要推力、Stroke、速度、Ball Screw/Lead Screw効率およびMoment Arm算定後に確定する。
# 5. Motor総数とMotor単体重量
Baseline:
```text
22ECT35 : 22 motors
22ECT48 : 20 motors
22ECT60 :  8 motors
Total   : 50 motors
```
Bare motor mass:
```text
22ECT35 : 22 × 67 g  = 1.474 kg
22ECT48 : 20 × 98 g  = 1.960 kg
22ECT60 :  8 × 123 g = 0.984 kg
--------------------------------
Total motor mass      ≈ 4.418 kg
```
Gear / Screw / Cable / Bearing / Driver / Harnessはこの重量に含まない。
# 6. Installed PowerとDuty
全Motorの最大連続機械出力を単純合算すると約2.5 kWとなるが、全Motorを同時に最大連続点で運転する設計とはしない。
```text
22ECT35 : 22 × 34 W = 748 W
22ECT48 : 20 × 54 W = 1080 W
22ECT60 :  8 × 86 W = 688 W
Total installed continuous mechanical rating ≈ 2516 W
```
Power System / Control SystemはGlobal Power Budgetを持ち、Priority、温度、Battery State、Safety Stateに応じて同時出力を制限する。
# 7. Energy Budget
Battery target:
```text
24 V nominal
50 Ah nominal
≈ 1.2 kWh nominal
usable ratio = 0.8
usable energy ≈ 0.96 kWh
```
通常5 hを保証するため:
```text
Average electrical power target <= 960 Wh / 5 h
                                <= 192 W
```
激しい運動45 minを保証するため:
```text
Average electrical power target <= 960 Wh / 0.75 h
                                <= 1280 W
```
したがって初期System Budgetは以下とする。
| Mode | Average electrical target |
| :- | --: |
| Standby / static | 50–100 W目標 |
| Nominal operation | 190 W以下を設計目標 |
| Intense operation | 1.2 kW以下を設計目標 |
| Short peak | 2.0 kW級をPower Systemで許容する設計候補 |
# 8. 伝達機構
Motorは直接Jointを駆動せず、筋肉模倣Actuatorとして以下を利用する。
- Ball Screw / Lead Screw
- Cable / Tendon
- Pulley
- Linkage
- Differential mechanism
出力関係:
```text
P = F v
τ = F r_eff
```
必要なForceと速度は相反するため、各筋肉GroupごとにScrew Lead / Gear Ratioを個別設定する。
# 9. 熱設計
34/54/86 Wは25°C条件の最大連続機械出力であり、Robot内部で同値を無条件に使用しない。
各Motorに温度監視または推定を持たせ、Thermal Deratingを行う。
Actuator Bundleでは隣接Motor間の熱干渉を評価する。
# 10. Power Driver
24 V BLDC Motor Driverを標準とする。
Driverは以下を満たすこと。
- Hall Sensor対応
- Current Control
- Current Limit
- Over Temperature Protection
- Command Timeout
- EmergencyStop入力
- Current / Voltage / Temperature Telemetry
# 11. 代替Motor
FAULHABER 2232 BX4（22 mm級、23 W）は首・小負荷Actuatorの代替候補とする。
Maxon ECX SPEED 22系は高出力密度が必要な箇所の比較候補とする。
Prototype 1の標準部品表にはPortescap 22ECTを用いる。
# 12. 未確定
- Screw type / lead
- Reduction ratio
- Actuator stroke
- Tendon material
- Tendon pretension
- Joint moment arm
- Peak duty duration
- Driver型番
- Battery chemistry / cell configuration
- Cooling
- Elbow extension actuator
- Hand actuator
