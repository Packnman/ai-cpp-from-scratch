# 腰 Dual Linear Actuator / Spine Lock 詳細設計

# 1. 目的

骨盤中央付近と左右肋骨側を結ぶ2本のLinear ActuatorによりWaist Pitch / Rollを生成し、中央Spineで主要荷重を支持する。

# 2. 機構

```text
 Left Rib                     Right Rib
    o                            o
     \                          /
      \ LA-L              LA-R /
       \                      /
        +---- Upper Torso ----+
               |
        Load-bearing Spine
          + Spine Lock
               |
             Pelvis
```

Linear Actuator両端には取付角変化を許容するSpherical / Clevis Joint等を設け、Actuatorへ大きな横荷重を与えない。

# 3. 運動

- 左右同相変位: 主にPitch成分
- 左右差動変位: 主にRoll成分
- 実際のPitch/Rollは取付幾何からForward/Inverse Kinematicsで算出する。

# 4. 力学

概算必要推力は `F ≈ τ / r_eff` を基準とするが、最終値は左右Actuatorの取付角を含むJacobianから計算する。

```text
tau_body = J(q)^T * F_linear
```

選定では最大推力だけでなく、Stroke、最大速度、連続推力、Peak推力、効率、Backdrivabilityを確認する。

# 5. Spine Lock

- 姿勢変更中: Unlock
- 静止保持: 条件成立後Lock
- Emergency時: Safety Sequenceに従いLock/Releaseを決定
- Lock成立確認Sensorを持つ

Spine Lockは上半身保持電力低減を目的とするが、Lock機構だけに全衝撃荷重を依存させない。

# 6. Safety

- Stroke hard limit
- Force / current limit
- 左右stroke difference limit
- actuator synchronization error
- spine lock failed / slip
- lateral load excessive

# 7. センサ

- LA-L stroke
- LA-R stroke
- motor currentまたはforce sensor
- torso IMU
- pelvis IMUまたは姿勢基準
- spine lock state

# 8. 未確定

- attachment geometry
- r_eff
- stroke
- force
- speed
- screw type
- reduction ratio
- lock mechanism
# 10. 採用Motor Baseline
左右2本のLinear Cylinderを使用する。
各Cylinderの初期Motor候補:
```text
Portescap 22ECT60 ×1
24 V
86 W max continuous mechanical rating @25°C
```
最終確定には以下を算定する。
```text
Required waist torque
Effective moment arm
Cylinder force
Stroke
Linear velocity
Screw efficiency
Thermal duty
```
必要推力を満たせない場合はMotor複数化またはScrew Lead変更を優先し、安易なMotor大型化は避ける。
