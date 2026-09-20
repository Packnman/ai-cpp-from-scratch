# CTRL-MOD-005 Posture Controller
# 責務
Body AttitudeとBalanceを安定化する。
# 入力
IMU / Joint / Contact / Force / Motion Target。
# 出力
- lower-body correction
- waist cylinder correction
- motion correction
# Priority
Fall preventionをtask trackingより優先する。
# Candidate
PD / LQR / MPC / Whole-body optimization。
