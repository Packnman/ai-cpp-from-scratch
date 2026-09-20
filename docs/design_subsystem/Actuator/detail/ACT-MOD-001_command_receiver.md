# ACT-MOD-001 Command Receiver 詳細設計

# 1. 目的
Control SystemおよびSafety Systemから指令を受信し、Command種別・優先度・Timestampを保持して後段へ渡す。

# 2. 入力
- DriveCommand
- Enable / Disable
- Stop / Reset
- SafeStop / EmergencyStop
- Safety Limit Update

# 3. 優先度
```text
EmergencyStop > SafeStop > SafetyLimit > Stop > DriveCommand
```

# 4. Interface
```cpp
class ICommandReceiver
{
public:
    virtual ~ICommandReceiver() = default;
    virtual void pushControlCommand(const ControlCommand&) = 0;
    virtual void pushSafetyCommand(const SafetyCommand&) = 0;
    virtual std::vector<ActuatorCommand> poll() = 0;
};
```

# 5. Queue
SafetyQueueとControlQueueを分離し、SafetyQueueを常に先に処理する。同一PriorityはFIFOとする。

# 6. Duplicate / Stale
Command IDで重複を検出し、同一Commandを二重実行しない。`now - timestamp > timeout` のCommandはStaleとしてRejectする。

# 7. Error
- malformed command
- unsupported schema
- duplicate command
- stale command
- queue overflow

# 8. Log
Command ID、Source、Type、Timestamp、Queue、Reject理由を記録する。

# 9. Test Point
- Safety priority
- FIFO
- duplicate
- timeout
- queue overflow
