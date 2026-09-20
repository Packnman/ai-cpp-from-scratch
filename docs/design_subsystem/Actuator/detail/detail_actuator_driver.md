# Actuator Driver 共通詳細設計

# 1. 目的
Motor Driver固有のPWM、Direction、Enable、Brake、Faultを共通Interfaceへ隠蔽する。

# 2. Driver責務
- configure
- enable
- disable
- command
- stop
- readState
- resetFault

# 3. PWM
PWM DutyはDriver詳細実装でCommandへ変換する。
```text
0.0 <= duty <= 1.0
```

# 4. Direction
方向反転時の瞬時大電流を防止するため、必要に応じてDutyを0へ戻した後にDirectionを切り替え、Dead Timeを設ける。

# 5. Enable
Enable前に以下を確認する。
- Power Available
- No Critical Fault
- Safety permits
- Driver configured

# 6. Brake
```cpp
enum class StopMode
{
    Coast,
    Brake,
    Hold,
    ControlledStop,
    EmergencyOff
};
```

# 7. Driver Fault
Driver固有Fault Codeを共通FaultTypeへ変換する。

# 8. Mock Driver
実機なしでposition response / velocity response / current rise / temperature rise / timeout / driver faultを模擬する。
