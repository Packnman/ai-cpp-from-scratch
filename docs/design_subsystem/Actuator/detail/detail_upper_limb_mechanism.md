# 上肢・肩甲骨 Tendon Drive 詳細設計

# 1. 目的

肩部に多数のMotorを置かず、胸・背中側MotorからTendon / Cableを介して肩甲骨と上腕骨を連動駆動し、軽量かつ広い作業域を得る。

# 2. 初期構成

```text
Chest Motor(s) ─┐
                 ├─ Tendon/Cable ─ Scapula Coupling ─ Humerus
Back Motor(s)  ──┘

Elbow: Local Rotary Motor
```

# 3. 設計方針

- Motor質量をTorsoへ集中する。
- Shoulder末端慣性を低減する。
- Scapula rotation / translationとHumerus motionを機械Linkまたは差動Cableで連動させる。
- Cable tensionはPushできないため、対向CableまたはSpring returnを使用する。

# 4. 状態

- drive motor position / velocity / current
- cable displacement
- cable tension
- scapula estimated pose
- humerus estimated pose
- elbow angle

# 5. Safety

- Over tension
- Slack
- Cable break suspicion
- Mechanical stop
- Shoulder collision

# 6. 未確定

- Motor数
- Cable routing
- Pulley radius
- Pretension
- Scapula/Humerus coupling ratio
- Shoulder reachable workspace
