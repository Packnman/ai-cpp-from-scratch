# Control System 結合テスト仕様
# 1. 目的
CTRL-MOD-001～010を接続し、Brain ActionからActuator Command、Feedback、Safety、ResultまでのControl Pipelineを検証する。
# 2. Test構成
```text
Fake Brain
→ Action Adapter
→ Motion Manager
→ Locomotion / Manipulation / Posture
→ Joint Controller
→ Mock Actuator
          ↑
Mock Sensor → Feedback Manager
Mock Safety → Safety Adapter
Mock Power  → Power Constraint
→ Execution Monitor
→ Fake Brain ActionResult
```
# 3. 正常系
## IT-CTRL-001 Walk
歩行Actionを投入し、Locomotion / Posture / Joint Controller経由でLower Body Muscle TargetとKnee Lock Commandが生成されること。
## IT-CTRL-002 Reach
Reach Actionを投入し、Shoulder / Scapular Muscle GroupとBiceps Targetが生成されること。
## IT-CTRL-003 Waist
Posture correctionによりDual Cylinder Targetが生成されること。
# 4. Hybrid Actuation
## IT-CTRL-010 Knee Lock
Gait Phaseに応じLock / Releaseが正しく切り替わること。
## IT-CTRL-011 Passive Ankle
AnkleへActive Commandを出さずSpring-Damper Stateを利用すること。
# 5. Safety
## IT-CTRL-020 Safety Limit
Velocity / Force Limitが全該当Controllerへ反映されること。
## IT-CTRL-021 SafeStop
Walking中SafeStopで安定停止Sequenceへ移行すること。
## IT-CTRL-022 EmergencyStop
Normal CommandよりEmergencyStopが優先されること。
# 6. Power Coordination
## IT-CTRL-030 Power Limited
Upper Body Task中にPower Limitが入った場合、Posture / Lower Body Priorityを維持しUpper BodyをDeratingすること。
# 7. Fault
- Sensor missing
- Actuator timeout
- control divergence
- posture unstable
- target unreachable
- communication loss
# 8. Realtime
- Cycle deadline
- overrun detection
- degraded mode
# 9. Acceptance
- ActionResultが一意に返る
- Safety Priorityが崩れない
- Faultが適切に伝播する
- Passive / Brake / Muscle / Linear各ActuationClassが正しく区別される
