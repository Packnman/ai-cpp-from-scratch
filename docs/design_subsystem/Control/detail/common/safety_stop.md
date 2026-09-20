# Safety / Stop Design
# 1. Priority
```text
EmergencyStop
> SafeStop
> SafetyLimit
> PowerLimit
> NormalControl
```
# 2. SafeStop
- gait deceleration
- stance stabilization
- waist stabilization
- upper-body motion reduction
- safe actuator stop
# 3. EStop
MuscleLike Actuatorはcurrent/force removal。
Knee LockはFail-safe designに従う。
Passive ankleはPower actionなし。
# 4. Restart
EmergencyStop後の自動Running復帰は禁止する。
