# CMP-MUS-001 Simulation Model

`SimMuscleActuator` は1台以上の `SimBLDCMotor` と `SimTransmission` をCompositionする。
初期Multi-Motor Modelは同一Command、同一Shaft速度、均等Load Sharingを仮定し、Motor torqueを合算する。

Joint torque要求は交換可能な `IMomentArmModel` により `F = tau_joint / r_eff` へ変換する。
初期実装は `ConstantMomentArmModel`、将来は関節角依存Geometry Modelへ交換可能である。
Tendon break時はPlant出力forceを0にし、overload、stroke limit、各Motor faultを共通Stateへ伝播する。

