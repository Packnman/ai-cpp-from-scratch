# ACT-MOD-008 Power Interface 詳細設計

# 1. 目的
Power SystemからActuator駆動可否と電源状態を取得し、Validator / Fault Monitor / Safety Limiterへ提供する。

# 2. PowerState
```cpp
struct PowerState
{
    bool available;
    double voltage;
    double currentLimit;
    BatteryState battery;
    std::optional<PowerFault> error;
    TimePoint timestamp;
};
```

# 3. Interface
```cpp
class IPowerInterface
{
public:
    virtual ~IPowerInterface() = default;
    virtual PowerState state() const = 0;
};
```

# 4. 駆動禁止条件
- Power Available false
- Under Voltage
- Over Voltage
- Power Supply Lost
- Critical Power Error

# 5. Current Limit
Power Systemが供給可能電流を低下させた場合、Safety Limiterへ動的制限として反映可能とする。

# 6. Stale
Power Stateが更新されない場合は安全側へ判定する。

# 7. Test Point
available / unavailable / undervoltage / current derating / stale
