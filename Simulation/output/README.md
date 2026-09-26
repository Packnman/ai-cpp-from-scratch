# Humanoid demo Viewer validation

Validation date: 2026-09-26

The Phase-1 humanoid demo was simulated for 2,000 physics steps (2.0 s) and
rendered through the MuJoCo Viewer path with a GLFW OSMesa context.

| Command | Goal | Target | Final position | Result |
| --- | --- | ---: | ---: | --- |
| `右手を上げて` | `RaiseRightArm` | -0.75 rad | -0.553761 rad | PASS |
| `右手を下げて` | `LowerRightArm` | 0.00 rad | -0.00674069 rad | PASS |

The raised right-arm pose also reached `-0.951068 rad` shoulder abduction and
`0.35787 rad` scapula rotation. The lower pose returned shoulder abduction to
`-0.0258725 rad`.

## Realtime validation

Run the VcXsrv-backed viewer and press Space to replay the coordinated scapula,
three-axis shoulder, and elbow motion:

```sh
./scripts/simulation/run.sh humanoid-viewer --command "右手を上げて"
```

The Simulation-labelled test suite passed: 10/10 tests.
