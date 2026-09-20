# CMP-MOT-001 Standard BLDC Motor Component
# 1. 概要
Portescap 22ECT35 / 22ECT48 / 22ECT60をPrototype 1のStandard Motor Familyとして定義する。
24 Vの高Torque Constant巻線をBaselineに固定する。
# 2. Prototype 1 Baseline
```text
S : 22ECT35 10B 80 01
M : 22ECT48 10B 35 01
L : 22ECT60 10B 21 01
```
# 3. 文書
- [要求仕様書](./requirement_spec.md)
- [設計仕様書](./design_spec.md)
- [詳細設計メイン](./detail/README.md)
# 4. 文書階層
```text
requirement_spec.md
→ design_spec.md
→ detail/README.md
   ├── motor_detailed_data.md
   ├── motor_variants.md
   ├── electrical_interface.md
   ├── mechanical_interface.md
   ├── thermal_design.md
   └── manufacturer_links.md
```
