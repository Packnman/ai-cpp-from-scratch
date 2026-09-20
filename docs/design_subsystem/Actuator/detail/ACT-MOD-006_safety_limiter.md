# ACT-MOD-006 Safety Limiter 詳細設計

# 1. 目的
DriveCommandおよび実状態に対してSafety Limitを適用する。

# 2. Limit種別
- Position
- Velocity
- Torque
- Current
- Temperature

# 3. Result
```cpp
struct LimitResult
{
    LimitDecision decision;
    DriveCommand command;
    std::vector<LimitViolation> violations;
};

enum class LimitDecision
{
    Pass,
    Clamp,
    Reject,
    SafeStop,
    EmergencyStop
};
```

# 4. Position
Mechanical Hard Limit越えはClampせずRejectまたはEmergencyStop対象とする。Software Limit内で安全に補正可能な場合のみClampする。

# 5. Velocity
Command Targetと実速度の両方を監視する。

# 6. Torque / Current
瞬時値だけでなく継続時間も判定条件にできる構成とする。

# 7. Temperature
```text
Normal → Warning → Limited → Stop
```

# 8. Dynamic Safety Limit
Safety Systemから受信したLimit値を動的に適用可能とする。

# 9. Interface
```cpp
class ISafetyLimiter
{
public:
    virtual ~ISafetyLimiter() = default;
    virtual LimitResult apply(
        const DriveCommand& command,
        const ActuatorDescriptor& descriptor,
        const ActuatorState& state,
        const SafetyState& safety
    ) const = 0;
};
```

# 10. Test Point
pass / clamp / reject / current limit / temperature derating / dynamic update


# 12. Hybrid機構のSafety

- Knee Brakeは支持中の意図しないReleaseを禁止する。
- Brake Lockは膝角速度が大きい状態で急激に掛けない。許容条件外ではControlled Lock Sequenceを使用する。
- Spine Lockも同様に姿勢・角速度・荷重条件を確認してLockする。
- Waist Linear ActuatorはStroke hard limit、推力limit、左右差limitを持つ。
- Tendon / Cableは最大張力、最低Pretension、断線検出を持つ。
- Passive Ankleは機械的角度limitと最大Spring Deflectionを監視する。
