# ACT-MOD-009 Log / Trace 詳細設計

# 1. 目的
DriveCommand、Validation、State、Safety Limit、Fault、Power、EmergencyStopを追跡可能にする。

# 2. TraceEvent
```cpp
struct ActuatorTraceEvent
{
    TraceId id;
    TimePoint timestamp;
    std::optional<CommandId> commandId;
    std::optional<ActuatorId> actuatorId;
    TraceType type;
    AttributeMap data;
};
```

# 3. 記録対象
- DriveCommand
- Validation Result
- Limited Command
- Driver Output
- ActuatorState
- SafetyCommand
- Fault
- PowerState
- ControlMode
- EmergencyStop
- Software Version
- Configuration Version

# 4. Correlation
```text
DriveCommand
→ Validation
→ Safety Limit
→ Driver Output
→ State
→ Result / Fault
```

# 5. Backend
- JSONL
- SQLite
- External Log Backend

# 6. Failure
Log失敗でActuator駆動を停止しない。ただしSafety Critical Log損失はWarningとして通知する。

# 7. Test Point
correlation / backend failure / timestamp / critical event
