# ACT-MOD-004 Drive Interface 詳細設計

# 1. 目的
制限済みCommandをPhysical Actuator / Motor Driverへ変換・出力する。

# 2. Interface
```cpp
class IActuatorDriver
{
public:
    virtual ~IActuatorDriver() = default;
    virtual void configure(const ActuatorDescriptor&) = 0;
    virtual void enable() = 0;
    virtual void disable() = 0;
    virtual void command(const DriveCommand&) = 0;
    virtual void stop(StopMode mode) = 0;
    virtual ActuatorState readState() = 0;
    virtual void resetFault() = 0;
};
```

# 3. Driver実装
```text
IActuatorDriver
├── MockActuatorDriver
├── PwmMotorDriver
├── CanMotorDriver
└── OtherDriver
```

# 4. Control Mode変換
Position / Velocity / Torque / Current / Stop / DisableをDriver固有出力へ変換する。

# 5. Enable / Disable
Disable中はCommandをDriverへ出力しない。

# 6. Driver Fault
Driver固有Fault Codeを共通ActuatorFaultへ変換する。

# 7. Test Point
enable / disable / mode conversion / stop / brake / driver fault


# 9. Driver種別拡張

```text
IActuatorDriver
├── IRotaryMotorDriver
├── ILinearActuatorDriver
├── IBrakeDriver
├── ITendonDriveDriver
└── IPassiveJointAdapter
```

`IPassiveJointAdapter` は駆動出力を持たず、角度・変位・反力等の状態取得を担当する。

Brake Driverは `lock()` / `release()` を明示的に扱う。
Linear Driverは Position/Velocity に加え Stroke / Force Limitを扱う。
Tendon Driverは Motor positionだけでなくCable tension / pretensionを状態として扱う。
