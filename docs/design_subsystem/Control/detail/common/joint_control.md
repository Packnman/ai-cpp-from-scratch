# Joint Control
# 1. MuscleLike
Joint / Body TargetをMuscle Group Force / Displacementへ配分する。
```text
tau_target
→ actuator geometry
→ force allocation
→ muscle group target
```
# 2. LinearCylinder
Waist Pitch / Roll Targetから左右Cylinder Length / Forceへ変換する。
# 3. BrakeControlled
Knee State / Gait PhaseからLock / Releaseを生成する。
# 4. PassiveElastic
AnkleへCommandは出さず、Spring-Damper ModelをControl Stateへ反映する。
# 5. Constraint
- force limit
- velocity limit
- stroke limit
- lock condition
- power limit
- thermal limit
