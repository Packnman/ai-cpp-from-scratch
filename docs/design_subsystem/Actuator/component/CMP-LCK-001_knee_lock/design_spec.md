# CMP-LCK-001 Knee Electromagnetic Lock Component 設計仕様書
# 1. Architecture
Passive Knee JointにElectromagnetic Brake / Lockを追加する。
# 2. State
Released / Engaging / Locked / Releasing / Fault。
# 3. Interlock
高Angular Velocity時のEngageを禁止する。
# 4. Detailed Design
- [Lock Mechanism](./detail/lock_mechanism.md)
- [Fail Safe](./detail/fail_safe.md)
- [Sensing and Control](./detail/sensing_control.md)
