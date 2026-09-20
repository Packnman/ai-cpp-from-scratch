# SIM-MOD-008 Viewer

`MuJoCoViewer` は GLFW と MuJoCo visualization API を使う任意ターゲットである。物理更新は `SimulationManager` に委譲し、描画周期と分離する。

- Space: pause/resume
- Right: paused single step
- Backspace: reset
- Mouse drag/wheel: camera
- Esc: close

headless CI は viewer を無効にし、同じ Plant/Manager を利用する。ビルド時は `AI_CPP_ENABLE_MUJOCO_VIEWER=ON` と GLFW 3 package が必要。
