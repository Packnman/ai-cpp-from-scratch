# Hybrid Actuation Model
# 1. Classes
- MuscleLike
- LinearCylinder
- BrakeControlled
- PassiveElastic
# 2. Body Mapping
```text
Neck          MuscleLike
Shoulder      MuscleLike
Elbow         MuscleLike
Waist         LinearCylinder
Hip / Thigh   MuscleLike
Knee          BrakeControlled + Passive
Ankle         PassiveElastic
Hand          TBD
```
# 3. State
Hybrid StateはContinuous StateとDiscrete Stateを同時に持つ。
```text
x_c = joint angle / velocity / body state / actuator state
q_d = knee lock state / controller state / support phase
```
# 4. Controller Implication
Knee Lockを含む歩行はHybrid State MachineまたはHybrid MPCで扱う。
