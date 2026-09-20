# PWR-DTL-002 Actuator Power Budget
# 1. 目的
筋肉模倣Actuator 50 MotorのInstalled RatingとMission上の同時Power Limitを分離して管理する。
# 2. Installed Rating
| Group | Motor構成 | Installed mechanical rating |
| :- | :- | --: |
| Neck | 22ECT35 ×4 | 136 W |
| Upper Body | 22ECT35 ×18 + 22ECT48 ×8 | 1.044 kW |
| Waist | 22ECT60 ×2 | 172 W |
| Lower Body | 22ECT48 ×12 + 22ECT60 ×6 | 1.164 kW |
| Total | 50 motors | 2.516 kW |
# 3. Mission Budget
| Mode | System average target | Duration target |
| :- | --: | --: |
| Nominal | <= 190 W | >= 5 h |
| Intense | <= 1.2 kW | >= 45 min |
| Short Peak | <= 2.0 kW class | transient only |
# 4. Budget Allocation
Control周期ごとにPower Managerは以下を評価する。
```text
P_available
= f(SOC, BatteryTemperature, BusVoltage, SafetyState, ThermalMargin)
```
Group request:
```text
P_req = Σ P_group_request
```
超過時:
```text
P_req > P_available
→ Priority順にClamp
→ Current Limit更新
→ ControlへPowerLimited通知
```
# 5. Priority
```text
P0 Safety / Control / Compute
P1 Lower Body stance / fall prevention
P2 Waist stabilization
P3 Upper Body task
P4 Neck / non-critical motion
```
# 6. Branch Limit
各Branchは少なくとも以下を持つ。
```text
enabled
priority
current_limit
power_limit
thermal_limit
fault_state
```
# 7. 運用上の注意
2.516 kWはMechanical Ratingの合算であり、Electrical Input設計値ではない。
Fuse、Driver、Cable、BMS電流値はMotor/Driverの最終Datasheet currentと実測Efficiencyにより確定する。
