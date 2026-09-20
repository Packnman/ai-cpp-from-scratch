# SIM-MOD-003 MuJoCo Model

`MuJoCoModel` は `mj_loadXML`/`mj_makeData` と対応する delete を RAII で対にする。コピーと move を禁止し、参照の寿命を所有者内に閉じる。初期化時に named joint/body/site/sensor/equality を logical ID table へ変換する。無名要素には安定した type-index 名を与える。

- Version: 3.13.0
- Failure: XML compiler error を例外メッセージへ含める
- Test: `UT-SIM-MOD-002_mujoco_plant.cpp` と全 MJCF smoke test
