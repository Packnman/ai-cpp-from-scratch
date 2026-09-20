# PWR-DTL-011 Test Design
# 1. Unit Test
## Battery Manager
- SOC boundary
- Low battery warning
- Temperature warning
- Invalid sensor state
## Distribution Manager
- Branch enable/disable
- Priority clamp
- Fault branch isolation
## Protection Manager
- Overcurrent
- Undervoltage
- Overvoltage
- Overtemperature
## Shutdown Manager
- Normal shutdown
- SafeShutdown
- EmergencyStop
# 2. Integration Test
- Lower Body high load + Upper Body command
- 2 kW class short transient simulation
- Battery SOC low during walking
- One Upper branch short fault
- One Lower branch current fault
- Logic Rail brownout immunity
- Communication isolation during actuator switching
# 3. Mission Test
## Nominal
```text
Duration >= 5 h
Average power <= 190 W target
No unexpected branch trip
```
## Intense
```text
Duration >= 45 min
Average power <= 1.2 kW target
Thermal limit obeyed
No Logic brownout
```
# 4. Fault Injection
- Current sensor stuck
- Voltage sensor invalid
- Temperature sensor high
- Branch switch fail-off
- Branch switch fail-on
- BMS fault
- Main contactor feedback mismatch
# 5. Acceptance
- Safety priority preserved
- Fault contained to intended branch where possible
- Required logs produced
- Restart policy obeyed
