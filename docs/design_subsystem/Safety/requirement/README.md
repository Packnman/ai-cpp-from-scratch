# Safety System 要求仕様書

# 1. 目的
本書は、人型ロボットシステムに搭載する Safety System が満たすべき要求事項を定義する。

Safety System は、Brain System および Control System から独立して安全状態を監視し、危険な指令または異常状態が発生した場合に制限、停止または非常停止を実行することを目的とする。

# 2. 関連文書

| 文書名                       | 内容                         |
| :------------------------ | :------------------------- |
| [システム要求仕様書](../../../design_system/requirement/README.md)                 | ロボットシステム全体の要求              |
| [システム設計仕様書](../../../design_system/spec/README.md) | システム構成およびSafety Systemの位置付け |

# 3. 適用範囲
- Joint Position監視
- Joint Velocity監視
- Torque監視
- Current監視
- Temperature監視
- Voltage監視
- Force監視
- Robot Attitude監視
- Communication監視
- Controller Status監視
- Sensor Status監視
- SafeStop
- EmergencyStop

# 4. 基本要求
- `REQ-SAFE-SYS-001` Safety System は Brain System から独立して動作可能であること。
- `REQ-SAFE-SYS-002` AI判断を安全判断の唯一の根拠としないこと。
- `REQ-SAFE-SYS-003` 危険な指令を制限または無効化可能であること。
- `REQ-SAFE-SYS-004` 必要に応じて Control System を停止可能であること。
- `REQ-SAFE-SYS-005` 必要に応じて Actuator System を非常停止可能であること。

# 5. 監視要求
- `REQ-SAFE-MON-001` Joint Positionを監視可能であること。
- `REQ-SAFE-MON-002` Joint Velocityを監視可能であること。
- `REQ-SAFE-MON-003` Torqueを監視可能であること。
- `REQ-SAFE-MON-004` Currentを監視可能であること。
- `REQ-SAFE-MON-005` Temperatureを監視可能であること。
- `REQ-SAFE-MON-006` Voltageを監視可能であること。
- `REQ-SAFE-MON-007` Forceを監視可能であること。
- `REQ-SAFE-MON-008` Robot Attitudeを監視可能であること。
- `REQ-SAFE-MON-009` Communication状態を監視可能であること。
- `REQ-SAFE-MON-010` Controller Statusを監視可能であること。
- `REQ-SAFE-MON-011` Sensor Statusを監視可能であること。

# 6. 安全レベル要求
- `REQ-SAFE-LVL-001` 正常状態を Level 0 として扱えること。
- `REQ-SAFE-LVL-002` 警告状態を Level 1 として扱えること。
- `REQ-SAFE-LVL-003` 制限動作状態を Level 2 として扱えること。
- `REQ-SAFE-LVL-004` 安全停止状態を Level 3 として扱えること。
- `REQ-SAFE-LVL-005` 非常停止状態を Level 4 として扱えること。
- `REQ-SAFE-LVL-006` 安全レベルに応じた処置を実行可能であること。

# 7. 制限要求
- `REQ-SAFE-LIM-001` 関節角を制限可能であること。
- `REQ-SAFE-LIM-002` 関節速度を制限可能であること。
- `REQ-SAFE-LIM-003` トルクを制限可能であること。
- `REQ-SAFE-LIM-004` 電流を制限可能であること。
- `REQ-SAFE-LIM-005` 接触力を制限可能であること。
- `REQ-SAFE-LIM-006` 温度異常時に出力を制限可能であること。

# 8. SafeStop要求
- `REQ-SAFE-STOP-001` 危険状態を検出した場合に SafeStop へ移行可能であること。
- `REQ-SAFE-STOP-002` SafeStop時に現在の危険なActionを停止すること。
- `REQ-SAFE-STOP-003` 必要に応じて安全姿勢へ移行可能であること。
- `REQ-SAFE-STOP-004` SafeStop状態を Brain System および HMI へ通知可能であること。

# 9. EmergencyStop要求
- `REQ-SAFE-EST-001` Emergency Stop入力を即時に処理可能であること。
- `REQ-SAFE-EST-002` Emergency Stop時に危険なActuator出力を停止すること。
- `REQ-SAFE-EST-003` Emergency Stop状態を明確に保持すること。
- `REQ-SAFE-EST-004` 意図しない自動復帰を禁止すること。
- `REQ-SAFE-EST-005` 復帰時に安全確認手順を要求可能であること。

# 10. 異常検出要求
- `REQ-SAFE-ERR-001` 通信異常を検出可能であること。
- `REQ-SAFE-ERR-002` センサ異常を検出可能であること。
- `REQ-SAFE-ERR-003` アクチュエータ異常を検出可能であること。
- `REQ-SAFE-ERR-004` 過電流を検出可能であること。
- `REQ-SAFE-ERR-005` 過熱を検出可能であること。
- `REQ-SAFE-ERR-006` 電源異常を検出可能であること。
- `REQ-SAFE-ERR-007` 転倒または異常姿勢を検出可能であること。
- `REQ-SAFE-ERR-008` 制御不能状態を検出可能であること。
- `REQ-SAFE-ERR-009` 計算処理異常を検出可能であること。

# 11. 通信断要求
- `REQ-SAFE-NET-001` 外部通信断によってSafety機能を停止しないこと。
- `REQ-SAFE-NET-002` 内部通信断時に対象サブシステムを安全状態へ移行可能であること。
- `REQ-SAFE-NET-003` 通信断時に危険な直前指令を保持し続けないこと。

# 12. 品質要求
- `REQ-SAFE-QUAL-001` Safety機能を個別に試験可能であること。
- `REQ-SAFE-QUAL-002` Safetyイベントをログへ記録可能であること。
- `REQ-SAFE-QUAL-003` Safety閾値を管理可能であること。
- `REQ-SAFE-QUAL-004` Safety設定値の変更履歴を管理可能であること。

# 13. 未確定事項
- 各安全閾値
- SafeStop方式
- EmergencyStop方式
- Watchdog構成
- Hardware Interlock構成
- Safety監視周期
