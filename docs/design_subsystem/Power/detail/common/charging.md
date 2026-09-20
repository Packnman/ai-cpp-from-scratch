# PWR-DTL-009 Charging
# 1. Charging State
```text
Disconnected
→ ChargerDetected
→ Precheck
→ Charging
→ Balancing
→ Complete
→ Disconnected
```
# 2. Interlock
Charging中はActuator Bus enableを原則禁止する。
Maintenance Mode等で例外を許可する場合は別Safety Requirementとする。
# 3. Charger Interface
- Charger present
- Charge voltage
- Charge current
- Battery temperature
- BMS charge_allowed
- charge fault
# 4. Protection
- Over-voltage
- Over-temperature
- Reverse connection
- Connector interlock
# 5. TBD
- Charger rating
- Charge time target
- External vs onboard charger
- Connector
