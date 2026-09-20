# CMP-DRV-001 BLDC Motor Driver Component 設計仕様書
# 1. Block
```text
24 V Bus
→ Fuse / Protection
→ 3-Phase Inverter
→ Motor
Hall → Commutation
Current Sense → Current Loop
MCU → Command / Telemetry / Fault
```
# 2. Control
Inner current loopをDriver内に持つ。
# 3. Interface
- command current/velocity
- enable
- telemetry current/voltage/temp
- fault
# 4. Detailed Design
- [Power Stage](./detail/power_stage.md)
- [Control Interface](./detail/control_interface.md)
- [Protection](./detail/protection.md)
