# Control Mode 共通詳細設計

# 1. 目的
Position / Velocity / Torque / Current / Stop / Disableの共通動作とMode切替条件を定義する。

# 2. Position
```text
Target Position
→ Position Controller
→ Velocity / Torque / Current Command
→ Driver
```

# 3. Velocity
```text
Target Velocity
→ Velocity Controller
→ Torque / Current Command
→ Driver
```

# 4. Torque
Torque SensorまたはCurrent推定をFeedbackとして利用可能とする。

# 5. Current
最下位電気制御LoopとしてMotor Currentを制御する。

# 6. Stop
Command生成を停止し、指定StopModeへ移行する。

# 7. Disable
Drive Outputを無効化する。

# 8. Mode切替条件
- supported mode
- actuator status
- target initialization
- integrator reset
- discontinuity prevention

# 9. Bumpless Transfer
Mode切替で出力が急変しないよう、初期値引継ぎまたは積分器Resetを行う。
