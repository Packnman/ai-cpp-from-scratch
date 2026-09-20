# Robot Plant Simulation

`Simulation/` は MuJoCo を用いた剛体 Robot Plant の実装です。Control と Actuator は MuJoCo API に依存せず、`IPlant` と `MuJoCoActuatorAdapter` を境界にします。

## Architecture and dependency

```text
Brain -> Control -> Actuator -> IPlant <- MuJoCoPlant -> MuJoCo
                              ^
                              +-- MuJoCoActuatorAdapter
```

Control、Actuator、Component は `mujoco.h` を include しません。Actuator の `component/sim` はモータ、スクリュー、腱、ばね、lock の応答を計算し、`Simulation/` は gravity、contact、constraint、rigid-body motion を計算します。この境界で motor dynamics の二重計算を防ぎます。

## Simulation loop

`SimulationManager` は simulation time だけを使い、既定で physics 1 kHz、Actuator 1 kHz、Control 100 Hz を実行します。各周期の callback が `IPlant` へ入力を設定し、`MuJoCoPlant::step` が effort を適用して `mj_step` を一回呼びます。viewer は 60 Hz を目安に描画しますが、physics の結果は FPS や wall clock に依存しません。

## Modules

各モジュールの責務と試験への対応は [detail/README.md](detail/README.md) を参照してください。

## Build

```sh
cmake -S . -B build -DAI_CPP_BUILD_SIMULATION=ON -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L simulation --output-on-failure
```

MuJoCo は `AI_CPP_MUJOCO_VERSION=3.13.0` に固定しています。システムに同版がなければ、既定で公式リポジトリから取得します。オフライン環境ではインストール済みパッケージを用意し、`-DAI_CPP_FETCH_MUJOCO=OFF` を指定してください。

## Run and headless

```sh
build/Simulation/simulation_cli --headless --model Simulation/model/humanoid/robot.xml --steps 1000
```

## Viewer

GLFW がある環境では `-DAI_CPP_ENABLE_MUJOCO_VIEWER=ON` でビューアを有効化できます。Space で pause/resume、Right で一ステップ、Backspace で reset、Esc で終了します。

```sh
cmake -S . -B build-viewer -DAI_CPP_BUILD_SIMULATION=ON -DAI_CPP_ENABLE_MUJOCO_VIEWER=ON
cmake --build build-viewer -j --target simulation_cli
build-viewer/Simulation/simulation_cli --viewer --model Simulation/model/humanoid/robot.xml
```

Linux の viewer build には X11 または Wayland の開発 package が必要です。GLFW 3.4 がなければ取得します。CI では viewer を OFF にして headless test を使用します。

## Test

`ctest -L simulation` は model/mapping/reset/time、state/contact/torque、multi-rate scheduling、Actuator feedback、Control→Actuator→Plant、gravity、解析解、determinism、headless を検証します。全モデルは CLI smoke test で MuJoCo compiler を通します。

## Folder structure

- `include/simulation`: MuJoCo 非依存の `IPlant` と Simulation 内部公開 API
- `src`: MuJoCo 所有・状態変換・接触・Actuator 接続・実行管理
- `model/test`: 単関節、二重振子、簡易脚
- `model/humanoid`: include 分割した全身モデルと asset 置場
- `config`: 実行・MuJoCo 設定例
- `test`: unit / integration テスト
- `requirement`, `spec`, `detail`: トレーサブルな設計文書

全身 MJCF の質量・慣性・寸法は明示的な仮値です。実機同定値への置換前に検証用途へ使用しないでください。
