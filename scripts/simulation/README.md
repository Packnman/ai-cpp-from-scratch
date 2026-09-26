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

# Brain command付きhumanoid viewer
./scripts/simulation/run.sh humanoid-viewer --command "右手を上げて"
```

humanoid viewerはSpaceで初期姿勢から同じ動作を再生し、Escで終了する。

### DockerからWindows版VcXsrvを使う

Windows PowerShellで、repository rootからVcXsrvを起動する。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\simulation\start-vcxsrv.ps1
```

初回のWindows Defender Firewall確認ではPrivate networkだけを許可する。その後、
container内で次を実行する。

```sh
./scripts/simulation/run.sh humanoid-viewer --command "右手を上げて"
```

`viewer` と `humanoid-viewer` は、不正または未設定の `DISPLAY` を
`host.docker.internal:0.0` に直し、VcXsrvのWGL-backed GLXを使う。別のdisplayを使う場合は
`AI_CPP_X11_DISPLAY` で上書きできる。

```sh
AI_CPP_X11_DISPLAY=host.docker.internal:1.0 ./scripts/simulation/run.sh humanoid-viewer
```

VcXsrvを手動でXLaunchから起動する場合は、display numberを`0`にし、
`Multiple windows`、`Start no client`、`Disable access control`を選ぶ。
VcXsrvが起動中なのに接続できない場合は、Windows FirewallのVcXsrv受信規則が
Private networkで許可されていることを確認する。

主な環境変数:

- `AI_CPP_SIMULATION_BUILD_DIR`: build directory
- `AI_CPP_JOBS`: build並列数。既定2
- `AI_CPP_CMAKE_GENERATOR`: CMake generator。既定Ninja
- `AI_CPP_SIMULATION_STEPS`: headless step数。既定1000
- `AI_CPP_SIMULATION_SMOKE_STEPS`: 各smoke modelのstep数。既定2
- `AI_CPP_X11_DISPLAY`: viewerのX11接続先。Dockerでは既定
  `host.docker.internal:0.0`
- `LIBGL_ALWAYS_INDIRECT`: GLX設定。VcXsrv/MuJoCo用の既定は`0`。
  `1`ではOpenGL 1.4に制限され、MuJoCoが起動しない

`test` と `sanitizer` の後ろの引数は `ctest` へ渡す。`headless` と
`viewer` の後ろの引数は `simulation_cli` へ、`humanoid-viewer` の後ろの引数は
`humanoid_demo` へ渡す。
