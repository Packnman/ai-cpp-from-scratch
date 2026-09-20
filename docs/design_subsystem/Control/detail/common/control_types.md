# Control Common Types
```cpp
enum class ActuationClass
{
    MuscleLike,
    LinearCylinder,
    BrakeControlled,
    PassiveElastic,
    TBD
};
enum class ActionState
{
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Timeout
};
struct MuscleGroupTarget
{
    std::string groupId;
    float targetForce;
    float targetDisplacement;
    float targetVelocity;
};
struct BrakeTarget
{
    std::string jointId;
    bool lock;
};
struct LinearTarget
{
    std::string actuatorId;
    float targetLength;
    float targetVelocity;
    float targetForce;
};
```
