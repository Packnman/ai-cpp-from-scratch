# 上肢・肩甲帯 筋肉模倣Actuation 詳細設計
# 1. 目的
人体の肩甲帯・上腕の筋骨格配置を参考に、小型BLDC Motorを複数連動させた筋肉模倣Actuatorで肩・肩甲骨・肘を駆動する。
# 2. Shoulder Joint Concept
肩回りはFree Joint構想とし、Clavicle、Scapula、Humerusを肩周囲機構として連動させる。
```text
Clavicle ─┐
Scapula  ─┼─ Shoulder Free Joint / Coupled Mechanism ─ Humerus
Humerus  ─┘
```
実機の拘束自由度、Joint Center、Scapula translation/rotation rangeはMechanical Detailで確定する。
# 3. Shoulder Actuator Groups
片側の初期構成:
| 模倣筋 | 役割の機械的対応 | Motor | 本数 |
| :- | :- | :- | --: |
| Deltoid | Shoulder cap方向の上腕駆動 | 22ECT35 | 3 |
| Pectoralis major | 前胸部からHumerus方向への牽引 | 22ECT48 | 2 |
| Serratus anterior | Rib側からScapula内側縁・下角方向への牽引 | 22ECT35 | 2 |
| Trapezius | Neck/Upper backからClavicle/Scapula方向への牽引 | 22ECT35 | 2 |
| Latissimus dorsi | Lower backからHumerus方向への牽引 | 22ECT48 | 2 |
筋肉模倣Actuatorは可能な限り人体の起始・停止方向を機械的Line-of-Actionへ写像する。
# 4. Serratus Anterior配置方針
前鋸筋は第1〜9肋骨側からScapulaの上角・内側縁・下角方向へ作用する人体解剖を参考にする。
Prototypeでは2 Motorを用いるBaselineとし、上部・中下部の作用を機械的に分担する案を優先検討する。
# 5. Elbow
肘のLocal Rotary Motor案は廃止する。
屈曲はBiceps brachii相当のMuscle-Like Actuatorで生成する。
```text
Biceps brachii : 22ECT35 ×2 / side
```
上腕二頭筋のScapula側起始からRadius側停止方向を参考にLine-of-Actionを設計する。
肘伸展側ActuatorはTBDとし、受動Return、Spring、またはTriceps相当Actuatorを比較する。
# 6. Neckとの境界
Sternocleidomastoid / Splenius capitisはNeck Moduleで扱い、Shoulder Actuatorと機械干渉しないRoutingとする。
# 7. State
- motor position / velocity / current
- actuator displacement
- tendon/cable displacement if used
- tendon/cable tension if used
- scapula estimated pose
- humerus estimated pose
- elbow angle
- actuator temperature / thermal estimate
# 8. Safety
- Over tension
- Slack
- Cable break suspicion
- Mechanical stop
- Shoulder collision
- Over current
- Thermal derating
- Asymmetric actuator fault
# 9. 未確定
- Exact origin/insertion bracket coordinates
- Cable routing / pulley radius
- Direct screw vs tendon transmission per muscle
- Pretension
- Scapula/Humerus coupling ratio
- Shoulder reachable workspace
- Elbow extension actuator
