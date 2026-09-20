# ACT-MOD-002 Command Validator 詳細設計

# 1. 目的
受信したCommandが対象Actuatorへ安全に適用可能かを検証する。

# 2. 検証順序
```text
Command Exists
↓
Actuator ID
↓
Control Mode
↓
Power State
↓
Actuator Status
↓
Timestamp / Timeout
↓
Target Range
↓
Safety Constraint
```

# 3. Result
```cpp
struct CommandValidationResult
{
    bool accepted;
    ValidationError error;
    std::optional<DriveCommand> command;
};
```

# 4. Rules
- RegistryにないActuator IDはReject
- `supportedModes` にないModeはReject
- `PowerAvailable == false` はReject
- Disabled / Fault / EmergencyStop中の通常CommandはReject
- Timeout済みCommandはReject
- Mechanical Hard Limit越えはReject

# 5. Interface
```cpp
class ICommandValidator
{
public:
    virtual ~ICommandValidator() = default;
    virtual CommandValidationResult validate(
        const DriveCommand& command,
        const ActuatorDescriptor& descriptor,
        const ActuatorState& state,
        const PowerState& power
    ) const = 0;
};
```

# 6. Test Point
invalid ID / unsupported mode / out-of-range / fault / power unavailable / timeout
