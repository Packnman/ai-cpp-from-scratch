# Mechanical Interface 詳細設計
# 1. Envelope
| Model | Diameter | Body Length | Mass |
| :- | --: | --: | --: |
| 22ECT35 | 22 mm | 35 mm | 67 g |
| 22ECT48 | 22 mm | 48 mm | 98 g |
| 22ECT60 | 22 mm | 60 mm | 123 g |
# 2. Common Drawing Features
22ECT35 / 48 drawingでは以下を確認済み。
- Body diameter: Ø22 ±0.1 mm
- Output shaft: Ø3 mm class
- Shaft extension: 10 mm class
- Front mounting: 3 × M2, 120° spacing, depth 3 mm minimum
- Cable length: 300 mm class
最終CAD設計ではManufacturer STP/DXFを直接使用する。
# 3. Bearing / Shaft Rule
Motor shaftへMuscle ActuatorのScrew axial thrustを直接入力しない。
```text
Motor Shaft
→ Flexible / Rigid Coupling
→ Transmission Input Shaft
→ Dedicated Axial/Radial Bearing
→ Screw / Pulley
```
# 4. Static Axial Limit
Manufacturer dataのMax Axial Static Force without Shaft Supportは45 N。
これはActuator Output Force設計値ではなくMotor shaftの制約として扱う。
# 5. CAD
Portescap Product PageからSTP / DXFを取得しProject CADのSource Geometryとする。
