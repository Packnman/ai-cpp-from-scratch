# CMP-TRN-001 Simulation Model

`SimTransmission` はMotor回転数 / Screw回転数として正の `gearRatio` を定義する。

```text
theta_screw = theta_motor / gearRatio
x = theta_screw * lead / (2 pi)
F = 2 pi * efficiency * tau_motor * gearRatio / lead
```

lead、efficiency、gearRatio、backlash、min/max positionはConfigurationで与える。
Screw選定値は未確定であり、Sourceに製品値として固定しない。Mechanical stopではpositionを制限し、外向きvelocityを0にする。

