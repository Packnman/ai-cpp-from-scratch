# CTRL-MOD-003 Locomotion Controller
# 責務
30 kg級Hybrid Lower Bodyの歩行・移動を制御する。
# 入力
- target pose / velocity
- ControlState
- contact
- power / safety constraints
# 出力
- footstep target
- body target
- hip/thigh muscle targets
- knee lock command
# Algorithm
LIPM / Capture Point / Foot Placement / MPCをBaseline候補とする。
Knee Lock採用時はHybrid MPCまたはHybrid State Machineを使用可能とする。
# Lower Body
Gluteus maximus / Rectus femoris / Biceps femoris / Adductor magnusを制御対象とする。
AnkleはPassive Spring-Damper Modelとして利用する。
