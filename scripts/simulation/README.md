# Simulation scripts

`run.sh` はMuJoCo/Actuator simulationのconfigure、build、test、実行をまとめる。
Repository外から呼び出しても動作する。

```sh
# Simulation/Actuator test
./scripts/simulation/run.sh test

# ASan/UBSan付きtest
./scripts/simulation/run.sh sanitizer

# simplified humanoidを1000 step実行
./scripts/simulation/run.sh headless

# 全4段階MJCFのload/step smoke test
./scripts/simulation/run.sh smoke

# GLFW viewer（LinuxではX11開発packageが必要）
./scripts/simulation/run.sh viewer
```

主な環境変数:

- `AI_CPP_SIMULATION_BUILD_DIR`: build directory
- `AI_CPP_JOBS`: build並列数。既定2
- `AI_CPP_CMAKE_GENERATOR`: CMake generator。既定Ninja
- `AI_CPP_SIMULATION_STEPS`: headless step数。既定1000
- `AI_CPP_SIMULATION_SMOKE_STEPS`: 各smoke modelのstep数。既定2

`test` と `sanitizer` の後ろの引数は `ctest` へ渡す。`headless` と
`viewer` の後ろの引数は `simulation_cli` へ渡すため、後置した
`--model` や `--steps` で既定値を上書きできる。
