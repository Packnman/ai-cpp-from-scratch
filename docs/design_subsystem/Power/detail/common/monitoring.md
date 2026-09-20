# PWR-DTL-005 Monitoring
# 1. Sampling対象
- Battery Voltage / Current
- 24 V Actuator Bus
- Logic Rail
- Sensor/Aux Rail
- Branch Current
- Battery Temperature
- Power Board Temperature
- Driver Temperature or thermal estimate
# 2. Sampling Tier
```text
Fast protection path  : hardware / driver internal
Control monitoring    : 100 Hz class
Telemetry / log       : 10–20 Hz class
Long-term trend       : 1 Hz class
```
最終Sampling RateはHardware選定後に確定する。
# 3. Derived State
- Electrical power
- Energy consumption
- SOC trend
- Group power
- Peak current history
- Thermal margin
# 4. PowerState拡張
```text
PowerState
├── battery
├── rails[]
├── branches[]
├── total_power
├── available_power
├── power_limit_state
├── warning
├── fault
└── timestamp
```
# 5. Sensor Fault
Current/Voltage Monitor fault時は値を0扱いせずUnknownとし、SafetyへFaultを通知する。
