# CMP-MOT-001 Standard BLDC Motor Component 設計仕様書
# 1. 採用Family
Prototype 1ではPortescap 22ECT Ultra EC 4-pole Slotless BLDC Familyを採用する。
# 2. Prototype 1 Baseline Part Number
筋肉Actuator用途では、高回転優先ではなく高Torque Constant・低Continuous Current側を優先する。
| Class | Baseline Article | Nominal V | Max continuous mechanical power @25°C | Max continuous torque | Max continuous current | Torque constant | Mass |
| :- | :- | --: | --: | --: | --: | --: | --: |
| S | 22ECT35 10B 80 01 | 24 V | 34 W | 19.5 mNm | 0.7 A | 27.31 mNm/A | 67 g |
| M | 22ECT48 10B 35 01 | 24 V | 54 W | 40.8 mNm | 1.5 A | 28.08 mNm/A | 98 g |
| L | 22ECT60 10B 21 01 | 24 V | 86 W | 64.3 mNm | 2.6 A | 25.97 mNm/A | 123 g |
# 3. 選定理由
- 3機種ともØ22 mm FamilyでMechanical Packagingを統一しやすい。
- 24 V Actuator Busへ直接適合させやすい。
- Torque Constantが約26～28 mNm/Aで近く、Current CommandからMotor Torqueへの変換を共通化しやすい。
- 低～中速で高Torqueを必要とするScrew / Tendon Actuationに適する。
- Hall Sensor内蔵によりPrototypeでCommutation / Speed Feedbackを簡潔に構成できる。
# 4. 重要な設計上の注意
`Max continuous mechanical power`、`Max continuous torque`、`No-load speed`は同一Operating Pointを表さない。
各最大値を同時成立する値としてTransmission設計へ使用してはならない。
実際のOperating PointはTorque-Speed Curve、Thermal Limit、Driver Limit、Transmission Efficiencyから決定する。
# 5. Mechanical
```text
Motor Body
├── Ø22 mm cylindrical envelope
├── Front mounting face
├── Ø3 mm class output shaft
├── Hall / Phase leads
└── Ball bearing support
```
# 6. Thermal
Maximum Winding Temperatureは125°C。
Robot内部では25°C Ratingをそのまま使用せず、Housing Temperature、Motor Bundle熱干渉、Duty Cycleを考慮してDeratingする。
# 7. 使用規則
- S / 22ECT35-80: Neck / Biceps / Deltoid / Serratus / Trapezius
- M / 22ECT48-35: Pectoralis / Latissimus / Rectus femoris / Biceps femoris / Adductor magnus
- L / 22ECT60-21: Gluteus maximus / Waist Cylinder
# 8. 公式製品情報
- 22ECT35: https://www.portescap.com/en/products/brushless-dc-motors/22ect35-ultra-ec-slotless-brushless-dc-motor
- 22ECT48: https://www.portescap.com/en/products/brushless-dc-motors/22ect48-ultra-ec-slotless-brushless-dc-motor
- 22ECT60: https://www.portescap.com/en/products/brushless-dc-motors/22ect60-ultra-ec-slotless-brushless-dc-motor
# 9. 詳細設計
- [Detailed Motor Data](./detail/motor_detailed_data.md)
- [Motor Variant / Selection](./detail/motor_variants.md)
- [Electrical Interface](./detail/electrical_interface.md)
- [Mechanical Interface](./detail/mechanical_interface.md)
- [Thermal Design](./detail/thermal_design.md)
- [Manufacturer Links / Documents](./detail/manufacturer_links.md)
