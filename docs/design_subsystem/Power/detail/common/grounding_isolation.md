# PWR-DTL-008 Grounding / Isolation
# 1. 目的
Motor switching noiseと大電流ReturnによるLogic reset / Sensor corruptionを防止する。
# 2. 基本方針
- Battery基準点は共通でもPower return pathを分離する。
- Motor Driver returnとLogic returnをDistribution pointで管理する。
- High-current cableとSensor/Communication cableを離す。
- Shield / chassis接続点を設計段階で固定する。
# 3. Isolation候補
- CAN / RS-485 isolation
- Isolated DC/DC for sensitive sensor
- Opto / digital isolator for EmergencyStop / branch control
必要箇所はEMI test結果で決定する。
# 4. Decoupling
各Driver近傍にLocal Bulk CapacitorとHigh-frequency bypassを配置する。
Main Actuator BusにもTransient吸収を持たせる。
# 5. Validation
- Maximum acceleration
- Direction reversal
- Simultaneous lower-body drive
- EStop
- Branch switching
でLogic Rail dipとCommunication errorを測定する。
