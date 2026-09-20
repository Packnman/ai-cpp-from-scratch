# 下肢 Hybrid Actuation 詳細設計

# 1. 構成

```text
Hip Pitch  : Active Motor
Hip Roll   : Active Motor
Knee       : Brake-Controlled Passive Joint
Ankle      : Passive Spring-Damper Joint
```

# 2. 設計目的

- Motor数削減
- distal mass削減
- 消費電力削減
- 倒立振子 / foot placement主体の歩行

# 3. 制御上の役割

Hip Pitch / Rollが主要な連続制御入力を生成する。
Knee Brakeは離散的なLock / Release入力として扱う。
Ankleは受動力学要素としてModelへ含める。

このため下肢はContinuous + Discrete + Passiveを含むHybrid Systemとして扱う。

# 4. Actuator System責務

- Hip torque/position command execution
- Knee brake command execution
- Knee/Ankle state reporting
- Safety limit / fault detection

歩行計画、Capture Point、MPC、Brake timing最適化そのものはControl System責務とする。
