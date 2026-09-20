# PWR-DTL-004 Power Distribution
# 1. 目的
Actuator FaultをBranch内へ封じ込め、Logic / Safetyを維持する。
# 2. Distribution Tree
```text
Battery
├── Main Fuse
├── Main Contactor
├── Logic DC/DC
├── Sensor/Aux DC/DC
└── Actuator Bus
    ├── Fuse + Switch : Neck
    ├── Fuse + Switch : Upper Left
    ├── Fuse + Switch : Upper Right
    ├── Fuse + Switch : Waist Left
    ├── Fuse + Switch : Waist Right
    ├── Fuse + Switch : Lower Left
    └── Fuse + Switch : Lower Right
```
# 3. Branch Switch
各Actuator BranchはElectronic SwitchまたはContactorを持つ。
必須機能:
- Remote enable
- Fault forced-off
- Current measurement
- Current limit coordination
- State feedback
# 4. Fuse / Breaker
最終定格はMotor Driver Datasheet、Wire Gauge、Peak Duty測定後に決定する。
FuseはWire protectionを主目的とし、Software Current Limitの代替としない。
# 5. Wiring
- Actuator trunk lineを左右Body側へ分配する。
- Logic wiringとMotor phase / high-current wiringを物理分離する。
- Return pathを明示する。
- Connectorには誤挿入防止を持たせる。
