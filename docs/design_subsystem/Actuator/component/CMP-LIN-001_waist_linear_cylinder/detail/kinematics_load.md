# Kinematics / Load 詳細設計
Torso GeometryからCylinder LengthとPitch/RollのJacobianを導出する。
Required force:
```text
F = J^-T * tau_body
```
Stroke / Speed / Screw Leadはこの計算結果から決定する。
