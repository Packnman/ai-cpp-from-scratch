# Simulation Module Detail

| Module | Responsibility | Detail |
|---|---|---|
| SIM-MOD-001 | scheduling and lifecycle | [SimulationManager](SIM-MOD-001_simulation_manager.md) |
| SIM-MOD-002 | engine-independent plant boundary | [MuJoCoPlant](SIM-MOD-002_mujoco_plant.md) |
| SIM-MOD-003 | MJCF ownership and ID mapping | [RobotModel](SIM-MOD-003_robot_model.md) |
| SIM-MOD-004 | Actuator output conversion | [ActuatorAdapter](SIM-MOD-004_actuator_adapter.md) |
| SIM-MOD-005 | state conversion | [StateAdapter](SIM-MOD-005_state_adapter.md) |
| SIM-MOD-006 | contact conversion | [ContactManager](SIM-MOD-006_contact_manager.md) |
| SIM-MOD-007 | ground-truth sensor view | [SensorSimulator](SIM-MOD-007_sensor_simulator.md) |
| SIM-MOD-008 | interactive visualization | [Viewer](SIM-MOD-008_viewer.md) |
| SIM-MOD-009 | CSV logging | [LogTrace](SIM-MOD-009_log_trace.md) |

各文書は対応する要件、公開 API、異常系、試験を記録する。上位仕様は [requirement](../requirement/simulation_requirement.md) と [system design](../spec/simulation_system_design_spec.md) を参照する。
