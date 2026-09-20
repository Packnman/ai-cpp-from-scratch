# PWR-MOD-001 Main Power Manager 詳細設計
# 1. 目的
Main Power State、Startup、Main Contactor、Main Power availabilityを管理する。
# 2. 責務
- Main Contactor制御
- Off / Starting / On / Warning / Limited / Fault / Shutdownの状態管理
- Startup precheck
- Main Power availability判定
- Shutdown ManagerからのMain Cut要求実行
- Main Contactor feedback監視
# 3. 非責務
- 他ModuleのSafety判断を代行しない。
- Motor commutationやMotor Driver内部保護はActuator/Driver側責務とする。
- Mechanical Safety LimitはSafety / Actuator Systemとの境界で扱う。
# 4. Input
```text
BatteryState
ProtectionState
ShutdownRequest
MainContactorFeedback
SystemStartRequest
```
# 5. Output
```text
PowerMode
MainPowerAvailable
MainContactorCommand
MainPowerFault
PowerEvent
```
# 6. 内部状態
```text
Off → Starting → On
On → Warning / Limited
Warning / Limited → On
On / Warning / Limited → Fault
Fault → Shutdown
Shutdown → Off
```
# 7. 基本処理
```text
update()
├── readInputs()
├── validate()
├── updateState()
├── evaluate()
├── emitCommandsOrState()
└── emitLog()
```
Module固有処理は上記Skeleton内で実装する。
# 8. Interface Concept
```cpp
class IMainPowerManager
{
public:
    virtual ~IMainPowerManager() =default;
    virtual void update() =0;
};
```
実Interfaceの型、戻り値、Error表現は共通`PowerState` / `PowerEvent`型と整合させる。
# 9. Error / Fault
- Invalid inputはFaultまたはUnknownとして明示する。
- Sensor値欠落を0として扱わない。
- Fault severityを保持する。
- Hardware protectionが作動した場合はSoftware Stateへ反映する。
# 10. Configuration
Module固有Threshold / Timing / LimitはHard-codeせずConfigurationで保持する。
# 11. Log / Trace
- State transition
- Warning / Fault transition
- Limit / Enable変更
- Configuration revision
を必要に応じ記録する。
# 12. 依存関係
Battery Manager、Protection Manager、Shutdown Manager、Distribution Manager
# 13. Testとの分離
Test Caseは本書に定義しない。
`../../test/PWR_TEST_DESIGN.md` から本Moduleの責務・Interfaceを参照して検証する。
# 14. TBD
- Hardware型番に依存する具体Threshold
- Sampling / Debounce / Timeoutの最終値
- Fault recoveryの実機条件
