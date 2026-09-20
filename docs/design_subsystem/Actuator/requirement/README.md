# Actuator System 要求仕様書

# 1. 目的
本書は、人型ロボットシステムに搭載する Actuator System が満たすべき要求事項を定義する。

Actuator System は、Control System から受け取った駆動指令に基づき、関節、ハンド、グリッパ、頭部、脚部その他の可動部を駆動し、ロボットの物理動作を実現することを目的とする。

# 2. 適用範囲
- Joint Motor
- Hand
- Gripper
- Leg
- Head
- Neck
- Other Actuator
- Actuator 状態監視
- Control System との連携
- Safety System との連携
- Power System との連携

# 3. 基本要求
- `REQ-ACT-SYS-001` Control System からの駆動指令を受信可能であること。
- `REQ-ACT-SYS-002` 指令された位置、速度、トルク等の目標に従って動作可能であること。
- `REQ-ACT-SYS-003` 各アクチュエータの現在状態を取得可能であること。
- `REQ-ACT-SYS-004` 各アクチュエータを個別に識別可能であること。
- `REQ-ACT-SYS-005` アクチュエータ異常を検出または上位システムへ通知可能であること。
- `REQ-ACT-SYS-006` Safety System からの停止または制限要求を優先可能であること。

# 4. 状態取得要求
- `REQ-ACT-STATE-001` Position を取得可能であること。
- `REQ-ACT-STATE-002` Velocity を取得可能であること。
- `REQ-ACT-STATE-003` Torque を取得可能であること。
- `REQ-ACT-STATE-004` Current を取得可能であること。
- `REQ-ACT-STATE-005` Temperature を取得可能であること。
- `REQ-ACT-STATE-006` Error 状態を取得可能であること。
- `REQ-ACT-STATE-007` 状態情報には必要に応じて Timestamp を付与可能であること。

# 5. 駆動要求
- `REQ-ACT-DRV-001` 位置指令を受け付け可能であること。
- `REQ-ACT-DRV-002` 速度指令を受け付け可能であること。
- `REQ-ACT-DRV-003` 必要に応じてトルクまたは電流指令を受け付け可能であること。
- `REQ-ACT-DRV-004` 動作開始および停止指令を受け付け可能であること。
- `REQ-ACT-DRV-005` 指令値が許容範囲外の場合、危険な動作を開始しないこと。
- `REQ-ACT-DRV-006` 駆動中に異常が発生した場合、上位システムへ通知可能であること。

# 6. 安全要求
- `REQ-ACT-SAFE-001` 非常停止指令を受けた場合、危険な駆動を停止すること。
- `REQ-ACT-SAFE-002` 許容最大位置を超える動作を防止可能であること。
- `REQ-ACT-SAFE-003` 許容最大速度を超える動作を防止可能であること。
- `REQ-ACT-SAFE-004` 許容最大トルクまたは電流を超える動作を防止可能であること。
- `REQ-ACT-SAFE-005` 過熱時に出力制限または停止可能であること。
- `REQ-ACT-SAFE-006` 通信断時に安全状態へ移行可能であること。

# 7. インタフェース要求
- `REQ-ACT-IF-001` Control System と駆動指令を送受信可能であること。
- `REQ-ACT-IF-002` Safety System から安全制御指令を受信可能であること。
- `REQ-ACT-IF-003` Power System から供給状態を取得可能であること。
- `REQ-ACT-IF-004` Actuator 状態を Control System および Safety System へ通知可能であること。
- `REQ-ACT-IF-005` アクチュエータ交換時に上位ロジックへの影響を最小限とすること。

# 8. エラー処理要求
- `REQ-ACT-ERR-001` 過電流を検出可能であること。
- `REQ-ACT-ERR-002` 過熱を検出可能であること。
- `REQ-ACT-ERR-003` センサ不整合を検出可能であること。
- `REQ-ACT-ERR-004` 指令応答異常を検出可能であること。
- `REQ-ACT-ERR-005` 回復不能な異常時に安全停止可能であること。

# 9. 品質要求
- `REQ-ACT-QUAL-001` 各アクチュエータを個別に試験可能であること。
- `REQ-ACT-QUAL-002` アクチュエータ状態をログへ記録可能であること。
- `REQ-ACT-QUAL-003` アクチュエータ種別および設定値を識別可能であること。
- `REQ-ACT-QUAL-004` アクチュエータ追加または交換に対応可能であること。

# 10. 未確定事項
- モータ種別
- 減速機構
- 最大関節角
- 最大速度
- 最大トルク
- 最大電流
- 温度上限
- 制御方式
- 通信方式
# Prototype 1 採用Component要求
- 筋肉模倣Active Actuatorはφ22 mm級BLDCを複数連動して構成する。
- 標準Motor FamilyはPortescap 22ECT35 / 22ECT48 / 22ECT60とする。
- Main Actuator Busは24 V nominalを基本とする。
- Motor単体ではなく、Screw / Cable / Tendon / Linkageを含むActuator Unitとして性能保証する。
- Robot Mass Targetは30 kgとする。
- 片腕で3 kg級の荷物を扱えることを目標とする。
- KneeはOptional Electromagnetic Lock、AnkleはPassive Spring-Damperを基本とする。
