# Electrical Interface 詳細設計
# 1. Bus
Prototype 1: 24 V nominal Actuator Bus。
# 2. Motor Interface
- 3-phase U/V/W
- Digital Hall A/B/C
- Hall electrical phasing: 120°
# 3. Selected Winding Parameters
| Parameter | 22ECT35-80 | 22ECT48-35 | 22ECT60-21 |
| :- | --: | --: | --: |
| Kt | 27.31 mNm/A | 28.08 mNm/A | 25.97 mNm/A |
| Ke | 2.86 V/krpm | 2.94 V/krpm | 2.72 V/krpm |
| R phase-phase | 9.2 Ω | 2.4 Ω | 1.08 Ω |
| L phase-phase | 0.75 mH | 0.24 mH | 0.123 mH |
| Continuous Current | 0.7 A | 1.5 A | 2.6 A |
# 4. Driver Parameterization
Driver configurationはMotor Classごとに以下を持つ。
```text
motor_class
nominal_voltage
torque_constant
back_emf_constant
phase_resistance
phase_inductance
continuous_current_limit
speed_limit
thermal_model
hall_phasing
```
# 5. Protection
- current limit
- over-speed
- under/over-voltage
- hall fault
- command timeout
- thermal derating
# 6. Harness
Phase leadとHall leadはNoise couplingを抑えるRoutingとし、Connector keyingを行う。
