# CTRL-MOD-007 Feedback Manager
# 責務
Sensor / Actuator / Power StateをControlStateへ統合する。
# Input
- IMU
- Joint State
- Contact
- Actuator State
- Power Limit
# Output
- timestamp-aligned ControlState
# Processing
Validation、timestamp alignment、filtering、missing-data flag。
