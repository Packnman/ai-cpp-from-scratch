# CMP-MOT-001 Standard BLDC Motor Component 要求仕様書
# 1. 目的
筋肉模倣ActuatorおよびWaist Linear Cylinderで共通利用する小型BLDC Motor Componentの要求を定義する。
# 2. 適用範囲
- Muscle-Like Actuator
- Waist Linear Cylinder
- 将来追加する小型Active Actuator
# 3. 機能要求
| ID | 要求 |
| :- | :- |
| REQ-CMP-MOT-FUN-001 | 24 V nominal Actuator Busで利用可能な巻線構成を選択可能であること |
| REQ-CMP-MOT-FUN-002 | Hall Sensorを利用したCommutation / Speed feedbackが可能であること |
| REQ-CMP-MOT-FUN-003 | Current Control対応Driverと組み合わせ可能であること |
| REQ-CMP-MOT-FUN-004 | S/M/Lの3出力Classを同一径Familyで構成できること |
| REQ-CMP-MOT-FUN-005 | ManufacturerがElectrical / Mechanical / Thermal Dataおよび2D/3D CADを提供していること |
# 4. 性能要求
| ID | 要求 |
| :- | :- |
| REQ-CMP-MOT-PER-001 | S Classは30 W級以上の最大連続機械出力を持つこと |
| REQ-CMP-MOT-PER-002 | M Classは50 W級以上の最大連続機械出力を持つこと |
| REQ-CMP-MOT-PER-003 | L Classは80 W級以上の最大連続機械出力を持つこと |
| REQ-CMP-MOT-PER-004 | Motor外径は22 mm級を基本とすること |
| REQ-CMP-MOT-PER-005 | 連続運転時はManufacturer thermal limitを超えないこと |
| REQ-CMP-MOT-PER-006 | Prototype 1では24 V巻線を優先し、低～中速高Torque用途に適したTorque Constantを持つこと |
# 5. Safety要求
- Overcurrent protectionをDriver側で持つこと。
- Winding temperatureを直接または推定で監視できること。
- Mechanical over-speedを禁止すること。
- ManufacturerのMaximum Winding Temperatureを超えないこと。
# 6. Interface要求
- 3-phase motor lead
- Digital Hall sensor
- Motor body mounting interface
- Shaft interface
# 7. Baseline Family
Prototype 1ではPortescap 22ECT35 / 22ECT48 / 22ECT60を標準Familyとする。
巻線型式の選定は設計仕様書で規定する。
# 8. 未確定
- Production connector
- Motor temperature sensor customization有無
- Final harness
- Driver正式型番
- Cooling condition
