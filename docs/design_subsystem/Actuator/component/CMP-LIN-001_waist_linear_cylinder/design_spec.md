# CMP-LIN-001 Waist Electric Linear Cylinder Component 設計仕様書
# 1. Structure
```text
22ECT60
→ Coupling
→ Screw
→ Nut / Slider
→ Rod
→ Spherical Joint
```
# 2. Pair Operation
左右Cylinder長差でRoll、共通伸縮でPitch成分を生成する。
# 3. State
length / velocity / current / force estimate / temperature / limits。
# 4. Detailed Design
- [Mechanical Stack](./detail/mechanical_stack.md)
- [Kinematics and Load](./detail/kinematics_load.md)
- [Sensing and Control](./detail/sensing_control.md)
