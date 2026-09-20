# Component Decision Log
# Baseline
- ComponentをActuator Systemから分離する。
- Portescap 22ECT35 / 48 / 60をStandard Motor Familyとする。
- Muscle ActuatorはStandard Motor + Transmission + SensorのComponent Assemblyとして扱う。
- Waist Linear Cylinderを独立Componentとする。
- Knee Electromagnetic Lockを独立Componentとする。
- Ankle Passive Spring-Damperを独立Componentとする。
# Pending
- Motor winding suffix
- Driver model
- Screw lead
- Tendon material
- Force sensor
- Knee lock mechanism
- Ankle k/c
# Motor Baseline Detailed Selection
- S: Portescap 22ECT35 10B 80 01
- M: Portescap 22ECT48 10B 35 01
- L: Portescap 22ECT60 10B 21 01
- Selection basis: 24 V, high torque constant, lower continuous current for muscle/transmission applications.

