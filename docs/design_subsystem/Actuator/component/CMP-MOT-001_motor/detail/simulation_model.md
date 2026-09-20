# CMP-MOT-001 Simulation Model

`SimBLDCMotor` は `V = Ri + L di/dt + Ke omega`、`tau = Kt i`、
`J d omega/dt = tau - tau_load - B omega` をSemi-Implicit Eulerで積分する。
Voltage、continuous current、maximum speedを各preset値で制限する。

| Preset | R | L | Kt | Ke | J | Current | Max speed |
| :- | --: | --: | --: | --: | --: | --: | --: |
| 22ECT35-80 | 9.23 ohm | 0.75 mH | 0.02731 Nm/A | 2.86 V/krpm | 3.6e-7 kg m2 | 0.7 A | 20000 rpm |
| 22ECT48-35 | 2.43 ohm | 0.24 mH | 0.02808 Nm/A | 2.94 V/krpm | 6.3e-7 kg m2 | 1.5 A | 20000 rpm |
| 22ECT60-21 | 1.11 ohm | 0.123 mH | 0.02597 Nm/A | 2.72 V/krpm | 8.71e-7 kg m2 | 2.6 A | 20000 rpm |

Viscous frictionは資料に値がないためpresetでは0とし、`MotorParameters` で設定可能とする。
Thermal Modelはcopper lossと資料のthermal resistance/time constantを用いる一次遅れである。

