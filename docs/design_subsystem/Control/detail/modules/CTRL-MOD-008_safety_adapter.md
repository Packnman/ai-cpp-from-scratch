# CTRL-MOD-008 Safety Adapter
# 責務
Safety Systemの制限・Stop CommandをControlへ適用する。
# Input
- velocity limit
- force / torque limit
- joint limit
- SafeStop
- EmergencyStop
# Output
- ControlConstraint
- immediate stop/derating command
# Priority
EmergencyStop > SafeStop > SafetyLimit > PowerLimit > NormalControl。
