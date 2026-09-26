# Prototype-1 Humanoid Model

このdirectoryは全身MJCFのsource of truthである。`model/demo/humanoid_supported.xml`
はpelvisをstandへ固定するviewer/test用wrapperで、body、actuator、sensor、contactの
実データはここへ集約する。

## Right-arm baseline

右腕は上肢設計書の確定済み構成を反映した最初のvertical sliceである。

| Muscle group | Motor baseline | Count | Simulated coordinate |
|---|---:|---:|---|
| Serratus anterior | 22ECT35 | 2 | `right_scapula_rotation` |
| Trapezius | 22ECT35 | 2 | `right_scapula_rotation` |
| Deltoid | 22ECT35 | 3 | `right_shoulder_abduction` |
| Pectoralis major | 22ECT48 | 2 | `right_shoulder_flexion` |
| Latissimus dorsi | 22ECT48 | 2 | `right_shoulder_rotation` |
| Biceps brachii | 22ECT35 | 2 | `right_elbow` |

肩を1つのMuJoCo ball jointではなく、肩甲骨回旋と肩3軸のscalar jointへ分解して
いる。これにより各Muscle-Like Actuatorをengine-independentな`IPlant`のjoint effort
へ個別に接続できる。`右手を上げて`は肩甲骨、肩3軸、肘の協調poseを生成する。

## Provisional parameters

設計書でTBDの値は、実値と誤認しないよう以下のsimulation baselineとして明示する。

| Parameter | Provisional value |
|---|---:|
| Scapula mass | 0.20 kg |
| Upper-arm mass / length | 0.90 kg / 0.30 m |
| Forearm mass / length | 0.65 kg / 0.28 m |
| Hand mass | 0.25 kg |
| Scapula moment arm | 0.025 m |
| Deltoid moment arm | 0.035 m |
| Pectoralis/Latissimus moment arm | 0.040 m |
| Biceps moment arm | 0.030 m |

Transmission lead、gear ratio、efficiency、strokeは現在
`defaultElbowMuscleConfig()`のsimulation defaultを使用する。これらはCAD/実機同定値
ではなく、制御経路と安全制限を検証するための仮値である。

## Replacement inputs

実機設計の確定後、次を置換する。

1. `body.xml`: CAD由来のlink寸法、質量、重心、慣性tensor、joint center/axis/range
2. `actuators.xml`とdriver config: bracket座標、routing、pulley半径、moment arm、stroke、pretension
3. `sensors.xml`: encoder、IMU、force/contact sensorの取付姿勢、周期、noise
4. `contact.xml`: self-collision除外、外装と足裏のfriction/contact parameter
5. `assets/`: 軽量化したvisual meshと独立したcollision mesh

参照設計:

- `docs/design_subsystem/Actuator/detail/detail_upper_limb_mechanism.md`
- `docs/design_subsystem/Actuator/detail/detail_commercial_motor_selection.md`
- `docs/design_subsystem/Actuator/component/CMP-MUS-001_muscle_actuator_unit/`
