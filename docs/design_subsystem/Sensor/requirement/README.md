# Sensor System 要求仕様書

# 1. 目的
本書は、人型ロボットシステムに搭載する Sensor System が満たすべき要求事項を定義する。

Sensor System は、周囲環境およびロボット内部状態を計測し、Brain System、Control System および Safety System が利用可能な情報として提供することを目的とする。

# 2. 適用範囲
- Camera
- IMU
- Distance Sensor
- Force Sensor
- Torque Sensor
- Touch Sensor
- Joint Angle Sensor
- Temperature Sensor
- Voltage Sensor
- Current Sensor

# 3. 基本要求
- `REQ-SEN-SYS-001` 外界情報を取得可能であること。
- `REQ-SEN-SYS-002` ロボット内部状態を取得可能であること。
- `REQ-SEN-SYS-003` センサデータを上位システムへ提供可能であること。
- `REQ-SEN-SYS-004` 各センサを一意に識別可能であること。
- `REQ-SEN-SYS-005` センサ異常を検出または通知可能であること。
- `REQ-SEN-SYS-006` 新しいセンサを追加可能な構成とすること。

# 4. 共通データ要求
各Sensor Dataは必要に応じて以下を保持可能とする。

| Field | 内容 |
| :- | :- |
| Sensor ID | センサ識別子 |
| Timestamp | 取得時刻 |
| Value | 計測値 |
| Status | センサ状態 |
| Error | 異常状態 |

- `REQ-SEN-DATA-001` Sensor IDを保持可能であること。
- `REQ-SEN-DATA-002` Timestampを保持可能であること。
- `REQ-SEN-DATA-003` Valueを保持可能であること。
- `REQ-SEN-DATA-004` Statusを保持可能であること。
- `REQ-SEN-DATA-005` Errorを保持可能であること。

# 5. Camera要求
- `REQ-SEN-CAM-001` 画像を取得可能であること。
- `REQ-SEN-CAM-002` 画像取得時刻を識別可能であること。
- `REQ-SEN-CAM-003` Camera状態を監視可能であること。
- `REQ-SEN-CAM-004` Brain Systemへ画像情報を提供可能であること。

# 6. IMU要求
- `REQ-SEN-IMU-001` 角速度を取得可能であること。
- `REQ-SEN-IMU-002` 加速度を取得可能であること。
- `REQ-SEN-IMU-003` 必要に応じて姿勢推定に利用可能な情報を提供すること。
- `REQ-SEN-IMU-004` Control SystemおよびSafety Systemへ情報提供可能であること。

# 7. 距離・空間センサ要求
- `REQ-SEN-DST-001` 周囲物体までの距離を取得可能であること。
- `REQ-SEN-DST-002` 障害物認識に利用可能な情報を提供すること。
- `REQ-SEN-DST-003` 空間認識に必要な精度および周期を満足可能であること。

# 8. 力・接触センサ要求
- `REQ-SEN-FRC-001` Forceを取得可能であること。
- `REQ-SEN-FRC-002` Torqueを取得可能であること。
- `REQ-SEN-FRC-003` Touch状態を取得可能であること。
- `REQ-SEN-FRC-004` Control SystemおよびSafety Systemへ力覚情報を提供可能であること。

# 9. 関節センサ要求
- `REQ-SEN-JNT-001` Joint Angleを取得可能であること。
- `REQ-SEN-JNT-002` 必要に応じてJoint Velocity算出に利用可能であること。
- `REQ-SEN-JNT-003` Control Systemへ関節状態を提供可能であること。
- `REQ-SEN-JNT-004` Safety Systemへ関節状態を提供可能であること。

# 10. 電気・温度センサ要求
- `REQ-SEN-PWR-001` Temperatureを取得可能であること。
- `REQ-SEN-PWR-002` Voltageを取得可能であること。
- `REQ-SEN-PWR-003` Currentを取得可能であること。
- `REQ-SEN-PWR-004` Power SystemおよびSafety Systemへ必要な情報を提供可能であること。

# 11. 異常検出要求
- `REQ-SEN-ERR-001` センサ未応答を検出可能であること。
- `REQ-SEN-ERR-002` 明らかな範囲外値を検出可能であること。
- `REQ-SEN-ERR-003` センサ異常を上位システムへ通知可能であること。
- `REQ-SEN-ERR-004` センサ異常時に古い値を現在値として無期限に使用しないこと。

# 12. 時刻要求
- `REQ-SEN-TIME-001` Sensor DataにTimestampを付与可能であること。
- `REQ-SEN-TIME-002` 複数センサ間の時間関係を識別可能であること。
- `REQ-SEN-TIME-003` センサデータ遅延を必要に応じて評価可能であること。

# 13. 品質要求
- `REQ-SEN-QUAL-001` 各センサを個別に試験可能であること。
- `REQ-SEN-QUAL-002` Sensor Dataを必要に応じてログへ記録可能であること。
- `REQ-SEN-QUAL-003` センサ型式および設定を識別可能であること。
- `REQ-SEN-QUAL-004` センサ交換時の上位システムへの影響を最小限とすること。

# 14. 未確定事項
- センサ型式
- 配置
- サンプリング周期
- 精度
- 分解能
- 通信方式
- 校正方式
