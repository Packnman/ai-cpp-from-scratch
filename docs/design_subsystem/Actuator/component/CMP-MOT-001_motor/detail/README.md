# CMP-MOT-001 Standard BLDC Motor Component 詳細設計 README
# 1. 目的
本書はCMP-MOT-001設計仕様書を実装・購入・CAD設計・Driver設計へ使用できる粒度へ展開する親文書である。
# 2. Prototype 1 Baseline
```text
S : Portescap 22ECT35 10B 80 01
M : Portescap 22ECT48 10B 35 01
L : Portescap 22ECT60 10B 21 01
Bus : 24 V nominal
Feedback : Digital Hall Sensors
```
# 3. 詳細文書
- [Detailed Motor Data](./motor_detailed_data.md)
- [Motor Variant / Selection](./motor_variants.md)
- [Electrical Interface](./electrical_interface.md)
- [Mechanical Interface](./mechanical_interface.md)
- [Thermal Design](./thermal_design.md)
- [Simulation Model](./simulation_model.md)
- [Manufacturer Links / Documents](./manufacturer_links.md)
# 4. Data Authority
Electrical / Mechanical / Thermal numeric dataはPortescap公式Product Page / Specification Pageを一次資料とする。
Robot固有のDerating値、Driver Limit、Transmission Operating Pointは本Projectの設計値として別管理する。
# 5. 実装原則
- Datasheet maximumを通常Operating Pointにしない。
- Continuous RatingはRobot内部温度条件で再Deratingする。
- Motor ShaftへScrew thrustを直接負担させず、Transmission側Bearingで支持する。
- 24 V Actuator Bus、Current-Controlled Driverを基本とする。
