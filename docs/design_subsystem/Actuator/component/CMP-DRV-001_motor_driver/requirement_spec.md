# CMP-DRV-001 BLDC Motor Driver Component 要求仕様書
# 1. 目的
Standard BLDC Motor Familyを安全に駆動するMotor Driver要求を定義する。
# 2. Functional
- 24 V nominal input
- 3-phase BLDC drive
- Hall commutation
- current control
- speed / current limit
- enable / disable
- telemetry
# 3. Protection
- over-current
- short-circuit
- under-voltage
- over-voltage
- over-temperature
- command timeout
# 4. Communication
Actuator ControllerとDigital Communication可能であること。
ProtocolはTBD。
# 5. Safety
EmergencyStop入力または上位Enable遮断でOutput Disable可能であること。
