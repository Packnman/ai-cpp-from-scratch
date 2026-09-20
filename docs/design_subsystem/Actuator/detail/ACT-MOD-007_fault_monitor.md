# ACT-MOD-007 Fault Monitor 詳細設計

# 1. 目的
Actuator State、Power State、Communication、Command応答を監視しFaultを検出する。

# 2. 監視対象
- OverCurrent
- OverTemperature
- PositionSensorError
- VelocitySensorError
- DriverError
- CommunicationError
- CommandTimeout
- PowerError
- ResponseError
- SensorMismatch

# 3. Fault
```cpp
struct ActuatorFault
{
    FaultId id;
    ActuatorId actuatorId;
    FaultType type;
    FaultLevel level;
    TimePoint timestamp;
    std::string description;
};
```

# 4. Fault Level
```text
0 Normal
1 Warning
2 Limited
3 Fault
4 Critical
```

# 5. 初期Mapping
| Fault | Level |
| :- | :- |
| Mild Temperature Warning | Warning |
| Current Limit approaching | Limited |
| Driver Fault | Fault |
| Safety Communication Lost | Critical |
| Severe OverCurrent | Critical |

# 6. Sensor Mismatch
Encoder / Mechanical Limit / Command / Secondary Sensor間の不整合を検出する。

# 7. Command Response
Command発行後一定時間State変化またはDriver応答がない場合Response Errorとする。

# 8. Interface
```cpp
class IFaultMonitor
{
public:
    virtual ~IFaultMonitor() = default;
    virtual std::vector<ActuatorFault> evaluate(
        const ActuatorState& state,
        const PowerState& power,
        const CommunicationState& communication
    ) = 0;
};
```

# 9. Test Point
overcurrent / overtemperature / communication loss / response timeout / mismatch / critical escalation


# 10. 追加Fault

- BrakeFailedToLock
- BrakeFailedToRelease
- LockSlip
- LinearStrokeLimit
- LinearForceOverload
- LinearActuatorMismatch
- TendonOverTension
- TendonSlack
- TendonBreakSuspected
- PassiveJointOverDeflection
- SpringDamperDegradation
