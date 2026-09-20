# CTRL-MOD-006 Joint Controller
# 責務
High-level targetをActuationClass別のActuator Commandへ変換する。
# Classes
- MuscleLike
- LinearCylinder
- BrakeControlled
- PassiveElastic
# MuscleLike
Target torque / motionからMuscle Group Force / Displacementへ配分する。
# LinearCylinder
Waist targetから左右Cylinder Targetを生成する。
# BrakeControlled
Knee Lock / Releaseを生成する。
# PassiveElastic
Commandなし。Model stateのみControllerへ返す。
