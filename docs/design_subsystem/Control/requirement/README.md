# Control System 要求仕様書

# 1. 目的
本書は、人型ロボットシステムに搭載する Control System が満たすべき要求事項を定義する。

Control System は、Brain System から受け取った Action を物理ロボットが実行可能な制御指令へ変換し、姿勢、歩行、関節、把持等の動作を実現することを目的とする。

# 2. 適用範囲
- Motion Control
- Locomotion Control
- Manipulation Control
- Posture Control
- Joint Control
- Actuator Control
- Action 実行状態管理
- Brain System との連携
- Safety System との連携

# 3. 基本要求
- `REQ-CTRL-SYS-001` Brain System から Action Command を受信可能であること。
- `REQ-CTRL-SYS-002` Action Command を実行可能な制御目標へ変換できること。
- `REQ-CTRL-SYS-003` アクチュエータへ制御指令を出力可能であること。
- `REQ-CTRL-SYS-004` ロボット状態を取得し制御へ利用可能であること。
- `REQ-CTRL-SYS-005` Action の実行結果を Brain System へ通知可能であること。
- `REQ-CTRL-SYS-006` Safety System の制約または停止要求を優先すること。

# 4. Motion Control要求
- `REQ-CTRL-MOT-001` 全身動作を協調して管理可能であること。
- `REQ-CTRL-MOT-002` 複数関節の目標を同時に生成可能であること。
- `REQ-CTRL-MOT-003` 動作開始、継続、停止を管理可能であること。
- `REQ-CTRL-MOT-004` 動作完了を判定可能であること。
- `REQ-CTRL-MOT-005` 動作失敗を判定可能であること。

# 5. Locomotion要求
- `REQ-CTRL-LOC-001` 指定方向または指定位置への移動を制御可能であること。
- `REQ-CTRL-LOC-002` 移動中に姿勢を維持可能であること。
- `REQ-CTRL-LOC-003` 移動速度を制限可能であること。
- `REQ-CTRL-LOC-004` 移動停止要求に応答可能であること。

# 6. Manipulation要求
- `REQ-CTRL-MAN-001` 腕および手の動作を制御可能であること。
- `REQ-CTRL-MAN-002` 指定対象への Reach 動作を実行可能であること。
- `REQ-CTRL-MAN-003` 把持および解放動作を実行可能であること。
- `REQ-CTRL-MAN-004` 力または接触情報を利用可能であること。

# 7. Posture Control要求
- `REQ-CTRL-POS-001` ロボットの姿勢を安定化可能であること。
- `REQ-CTRL-POS-002` 姿勢異常を検出可能であること。
- `REQ-CTRL-POS-003` 外乱に対して姿勢を維持または回復可能であること。
- `REQ-CTRL-POS-004` 安全姿勢への移行指令を受け付け可能であること。

# 8. Joint Control要求
- `REQ-CTRL-JNT-001` 各関節の位置制御が可能であること。
- `REQ-CTRL-JNT-002` 各関節の速度制御が可能であること。
- `REQ-CTRL-JNT-003` 必要に応じてトルクまたは電流制御が可能であること。
- `REQ-CTRL-JNT-004` 関節制限を超える指令を抑制可能であること。
- `REQ-CTRL-JNT-005` 関節状態を上位制御へ提供可能であること。

# 9. リアルタイム性要求
- `REQ-CTRL-RT-001` 各制御ループに必要な処理周期を満足すること。
- `REQ-CTRL-RT-002` 上位処理遅延が下位制御の即時停止につながらない構成とすること。
- `REQ-CTRL-RT-003` 制御周期超過を検出可能であること。
- `REQ-CTRL-RT-004` 制御周期異常時に安全側へ遷移可能であること。

# 10. Safety連携要求
- `REQ-CTRL-SAFE-001` Safety System からの速度制限を反映可能であること。
- `REQ-CTRL-SAFE-002` Safety System からのトルク制限を反映可能であること。
- `REQ-CTRL-SAFE-003` SafeStop 指令を受け付け可能であること。
- `REQ-CTRL-SAFE-004` EmergencyStop 指令を受け付け可能であること。
- `REQ-CTRL-SAFE-005` Safety System の指令を Brain System の通常指令より優先すること。

# 11. エラー処理要求
- `REQ-CTRL-ERR-001` 制御不能状態を検出可能であること。
- `REQ-CTRL-ERR-002` アクチュエータ応答異常を検出可能であること。
- `REQ-CTRL-ERR-003` センサ異常によって安全な制御ができない場合に安全側へ遷移可能であること。
- `REQ-CTRL-ERR-004` Action失敗理由を Brain System へ通知可能であること。

# 12. 品質要求
- `REQ-CTRL-QUAL-001` 各制御機能を個別試験可能であること。
- `REQ-CTRL-QUAL-002` 実機なしで一部制御を模擬可能であること。
- `REQ-CTRL-QUAL-003` 制御状態をログへ記録可能であること。
- `REQ-CTRL-QUAL-004` 制御パラメータをバージョン管理可能であること。

# 13. 未確定事項
- 制御周期
- 制御方式
- Motion Planning方式
- Walking方式
- 姿勢制御方式
- Joint Control方式
- Actuator Control方式
