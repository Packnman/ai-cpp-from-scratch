# CTRL-MOD-002 Motion Manager
# 責務
全身MotionのSequenceとController協調を管理する。
# 入力
- NormalizedAction
- ControlState
- SafetyLimit
- PowerLimit
# 出力
- LocomotionTarget
- ManipulationTarget
- PostureTarget
- Cancel / Stop
# State
Idle / Preparing / Running / Stopping / Failed。
# 処理
Action分解、Controller選択、複数部位同期、Completion条件管理。
