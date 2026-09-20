# Brake-Controlled Joint 詳細設計

# 1. 目的

Motorで関節角を能動生成せず、Lock / Releaseによって関節の運動自由度を切り替える。初期対象はKneeおよびSpine Lock。

# 2. Knee基本動作

```text
Swing Phase   : Release
Pre-contact   : Knee state監視
Support Phase : Lock
```

# 3. 必須Sensor

- joint angle
- angular velocity
- lock state
- brake temperature（必要に応じ）

# 4. Fail-safe

膝用途では無励磁保持型Brakeを第一候補とし、Power loss時の意図しない膝折れを抑える。

ただしEmergencyStop時に即Lockすることが常に安全とは限らないため、角速度・接地状態をSafety Sequenceで判定する。

# 5. Lock条件

- joint velocity below threshold または Controlled Lock許可
- mechanical angle within allowed range
- no brake fault
- Safety state permits

# 6. Fault

- failed to lock
- failed to release
- lock slip
- overheating
- state sensor disagreement
