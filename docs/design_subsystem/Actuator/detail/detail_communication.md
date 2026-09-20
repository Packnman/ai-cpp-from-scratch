# Actuator Communication 共通詳細設計

# 1. 目的
Control / Safety / Power / DriverとのCommand / State通信形式、Timeout、Retry、Integrityを定義する。

# 2. Command Format
```text
Command ID
Actuator ID
Command Type
Control Mode
Target
Limit
Timestamp
Timeout
Version
Integrity
```

# 3. State Format
```text
Actuator ID
Position
Velocity
Torque
Current
Temperature
Status
Fault
Timestamp
Sequence
Version
Integrity
```

# 4. Timeout
- Control Command
- Driver Response
- State Update
- Safety Link
- Power State

# 5. Retry
DriveCommandは無条件再送しない。Command IDで重複実行を防止する。

# 6. Integrity
通信方式に応じChecksum / CRCを利用する。

# 7. Sequence
Sequence番号でduplicate / missing / out-of-orderを検出可能とする。

# 8. Communication Loss
Control断ではHold / ControlledStop / SafeStopのいずれかへ移行する。Safety断はSafety Faultとして扱い、Safety Levelを上げる。
