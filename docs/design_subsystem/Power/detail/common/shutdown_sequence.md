# PWR-DTL-007 Shutdown Sequence
# 1. Normal Shutdown
```text
Brain shutdown request
→ Control motion stop
→ Actuator zero/hold command
→ Branch disable
→ Log flush
→ Sensor/Aux optional off
→ Logic shutdown
→ Main contactor off
```
# 2. SafeShutdown
```text
Power Warning
→ Control stop request
→ Lower Body stable-state request
→ Waist stabilize / lock if safe
→ Upper Body power reduction
→ Actuator stop confirmation
→ Branch off
→ Log flush
→ Main shutdown
```
# 3. EmergencyStop
EmergencyStopはMotion Safetyの要求を優先する。
Power Systemは一律全断せず、Actuator特性ごとにSafety Systemの遮断Commandに従う。
- Active muscle actuator: torque/current removal
- Waist cylinder: drive removal after safe mechanical state if available
- Knee electromagnetic lock: fail-safe designに従う
- Passive ankle: no power action
- Logic / Safety:必要時間維持
# 4. Restart
EmergencyStop / severe Power Fault後は自動Running復帰を禁止する。
Manual acknowledge、Fault clear、Power self-testを要求する。
