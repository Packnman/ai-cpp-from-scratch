# PWR-DTL-003 Battery Pack
# 1. Baseline
```text
Nominal voltage  : 24 V class
Capacity         : 50 Ah class
Nominal energy   : ≈ 1.2 kWh
Usable ratio     : 0.8
Usable energy    : ≈ 0.96 kWh
```
# 2. BMS機能
- Cell over-voltage
- Cell under-voltage
- Pack over-current
- Short-circuit protection
- Charge over-current
- Battery temperature monitoring
- SOC estimation
- SOH estimation
- Contactor control
- Fault output
# 3. Pack Interface
```text
BatteryPackState
├── pack_voltage
├── pack_current
├── soc
├── soh
├── min_cell_voltage
├── max_cell_voltage
├── min_temperature
├── max_temperature
├── charge_state
├── discharge_allowed
├── charge_allowed
└── fault
```
# 4. Capacity Rule
Nominal Mission:
```text
0.96 kWh / 5 h = 192 W
```
Intense Mission:
```text
0.96 kWh / 0.75 h = 1.28 kW
```
設計目標はそれぞれ190 W以下、1.2 kW以下とする。
# 5. TBD
- Chemistry
- Cell model
- Series / parallel configuration
- BMS model
- Continuous / peak pack current
- Mechanical enclosure
- Connector
