# ACT-MOD-005 State Monitor 詳細設計

# 1. 目的
ActuatorおよびDriverから状態を取得し共通ActuatorStateへ変換する。

# 2. 監視項目
- Position
- Velocity
- Torque
- Current
- Temperature
- Driver Status
- Error
- Timestamp

# 3. Interface
```cpp
class IStateMonitor
{
public:
    virtual ~IStateMonitor() = default;
    virtual ActuatorState sample(
        IActuatorDriver& driver,
        const ActuatorDescriptor& descriptor
    ) = 0;
};
```

# 4. Sampling
Control Cycleと独立設定可能とし、測定時刻に近いTimestampを付与する。

# 5. Stale
一定時間State更新がない場合はStaleとしてFault Monitorへ通知する。

# 6. Estimation
VelocityはEncoder差分、TorqueはTorque SensorまたはMotor Current等から推定可能とする。

# 7. Filter
Current / Temperature等へFilterを適用可能とする。Safety用Raw値とControl用Filtered値は分離可能とする。

# 8. Test Point
sampling / timestamp / stale / velocity / torque / filter


# 10. 機構別状態

共通Stateに加え以下を取得可能とする。

- Knee: angle, angular velocity, brake lock state
- Ankle: angle, spring deflection, estimated spring torque
- Waist: left/right stroke, linear velocity, estimated force, spine lock state
- Shoulder Tendon: motor position, cable displacement, cable tension, scapula/humerus estimated pose
