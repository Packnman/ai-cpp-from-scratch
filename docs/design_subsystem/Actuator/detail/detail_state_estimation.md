# State Estimation 共通詳細設計

# 1. 目的
生Sensor値からActuator Stateとして利用するVelocity、Torque、Current、Temperature等を生成する。

# 2. Velocity
```text
velocity = (position[k] - position[k-1]) / dt
```
Noiseが大きい場合はFilterを利用する。

# 3. Torque
候補:
- Torque Sensor
- Current × Kt
- Observer
- Mechanical Model

初期実装では利用可能Sensorを優先する。

# 4. Current
Driver Current Sensor値を取得し、必要に応じLow Pass Filterを適用する。

# 5. Temperature
Motor / Driver Temperatureを個別に取得可能とする。

# 6. Filter
FilterはSafety検出遅延を増大させないこと。Safety用Raw値とControl用Filtered値を分離可能とする。

# 7. Sensor Fusion
複数Sensorを使う場合はSource、Timestamp、Confidenceを保持する。

# 8. Sensor Mismatch
冗長Sensor間の差が閾値を超えた場合Fault Monitorへ通知する。
