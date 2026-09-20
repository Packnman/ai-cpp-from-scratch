# CMP-LCK-001 Simulation Model

`SimKneeLock` はReleased / Engaging / Locked / Releasing / Faultを持つ。
許容角速度を超えるLock要求とlock failure注入はFaultへ遷移する。
Locked時は無限剛性で関節を直接固定せず、lock位置、有限stiffness、holding torque limitをPlantへ通知する。

Engage/Release時間、最大Engage角速度、保持Torque、Stiffnessは実機方式未確定のためConfiguration値とする。

