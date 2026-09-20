# SIM-MOD-001 Simulation Manager

`SimulationManager` は `IPlant` を所有し、initialize/reset/pause/runSteps を管理する。物理周期を基準に、Actuator と Control callback をそれぞれの accumulator が期限に達した時だけ呼ぶ。壁時計を計算結果へ混入させない。

- Input: `SimulationTiming`, callbacks
- Output: `IPlant::step(physicsDt)`
- Error: timestep が有限な正値でなければ構築を拒否
- Test: `UT-SIM-MOD-001_simulation_manager.cpp`
