# Simulation System Design Specification

## Architecture

```text
Control callback -> ActuatorManager / IActuatorDriver
                 -> PlantOutput
                 -> MuJoCoActuatorAdapter
                 -> IPlant
                 -> MuJoCoPlant -> MuJoCo C API / MJCF
                                      |
                                      +-> StateAdapter / ContactManager
```

`IPlant` と `PlantTypes` はエンジン非依存である。MuJoCo の `mjModel`/`mjData` は `MuJoCoModel` が単独所有し、Simulation 外へライフタイムを渡さない。Actuator 内の `component/sim` はモータ・直動・受動要素のモデルだけを担当し、剛体 world/plant は Simulation が担当する。

## Execution

`SimulationManager::step()` は control、actuator callback の期限をシミュレーション時刻で判定した後、固定 `physicsDt` で Plant を一回進める。pause 中は時刻を進めない。reset は Plant と全 accumulator を初期状態へ戻す。

## State and mapping

モデル読み込み時に joint/body/site/sensor/equality の名前を logical ID として ID/address table を生成する。状態変換は qpos/qvel/qacc/qfrc、body pose/velocity、contact wrench、sensor data を値型へコピーする。

## Errors

モデル読込失敗、未知 ID、非有限または非正 timestep、負の拘束係数は例外とする。未定義 equality constraint は同名 joint に有限 PD torque を適用する。

## Model policy

`model/test` で自由度を段階的に増やし、`model/humanoid/robot.xml` は body/actuator/sensor/contact を include する。mesh/texture は専用 asset directory に置く。現時点の全身値は placeholder であり、コメントを外さない。
