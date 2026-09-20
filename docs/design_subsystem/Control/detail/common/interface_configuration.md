# Interface / Configuration
```cpp
class IController
{
public:
    virtual ~IController() =default;
    virtual void initialize() =0;
    virtual void setTarget() =0;
    virtual void update() =0;
    virtual void stop() =0;
    virtual void reset() =0;
};
```
Config例:
```text
ControlConfig
├── control_cycle
├── locomotion
├── posture
├── manipulation
├── muscle_groups[]
├── waist_geometry
├── knee_lock
├── ankle_model
├── safety_limits
└── power_limits
```
