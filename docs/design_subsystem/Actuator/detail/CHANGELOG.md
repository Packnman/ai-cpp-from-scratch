# Actuator Design 改訂内容

今回の改訂で以下を設計へ反映した。

- Actuator種別をMotor中心からHybrid Actuationへ拡張
- Shoulder: Torso motor + Tendon/Cable + Scapula/Humerus coupling
- Elbow: Local rotary motor [SUPERSEDED: Biceps brachii muscle-like actuator]
- Waist: Dual linear actuator + load-bearing spine + spine lock
- Hip: Active Pitch/Roll
- Knee: Electromagnetic brake-controlled joint
- Ankle: Passive spring-damper
- Actuator Manager / Drive / State / Safety / Faultの各詳細設計をHybrid機構へ対応
- Linear / Brake / Passive Elastic / Tendon機構の専用詳細設計を追加

# Current Baseline Update
- Neck: Sternocleidomastoid + Splenius capitis muscle-like actuators
- Shoulder: Free-joint concept with Deltoid / Pectoralis major / Serratus anterior / Trapezius / Latissimus dorsi
- Elbow: Biceps brachii muscle-like actuator; local rotary motor removed
- Hip/Thigh: Gluteus maximus / Rectus femoris / Biceps femoris / Adductor magnus muscle-like actuators
- Knee: Passive joint + Optional electromagnetic lock
- Ankle: Passive spring-damper
- Waist: Dual electric linear cylinders + load-bearing spine + lock
