# CMP-TRN-001 Muscle Transmission Component 設計仕様書
# 1. Selection Rule
```text
High force / lower speed → screw lead小
Higher speed / lower force → screw lead大
Remote actuation → tendon/cable
Complex line-of-action → pulley/linkage
```
# 2. Equation
```text
P_out = F * v
tau_motor → transmission → F_linear
```
# 3. Detailed Design
- [Screw Drive](./detail/screw_drive.md)
- [Tendon Cable](./detail/tendon_cable.md)
- [Pulley Linkage](./detail/pulley_linkage.md)
