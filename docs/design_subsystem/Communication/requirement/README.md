# Communication System 要求仕様書

# 1. 目的
本書は、人型ロボットシステムに搭載する Communication System が満たすべき要求事項を定義する。

Communication System は、ロボット内部サブシステム間および外部計算機との間で、指令、状態、認識結果、エラー等を確実に送受信することを目的とする。

# 2. 適用範囲
- 内部通信
- 外部通信
- 通信状態監視
- 再接続
- Timeout
- Message 識別
- Error 通知
- 時刻同期補助

# 3. 基本要求
- `REQ-COM-SYS-001` サブシステム間で必要な情報を送受信可能であること。
- `REQ-COM-SYS-002` 外部計算機と通信可能であること。
- `REQ-COM-SYS-003` 通信状態を監視可能であること。
- `REQ-COM-SYS-004` 通信断を検出可能であること。
- `REQ-COM-SYS-005` 通信復旧を検出可能であること。
- `REQ-COM-SYS-006` 通信障害が安全機能を無効化しないこと。

# 4. 内部通信要求
- `REQ-COM-INT-001` Sensor Data を送信可能であること。
- `REQ-COM-INT-002` Robot State を送信可能であること。
- `REQ-COM-INT-003` Action Command を送信可能であること。
- `REQ-COM-INT-004` Control Command を送信可能であること。
- `REQ-COM-INT-005` Action Result を送信可能であること。
- `REQ-COM-INT-006` Error 情報を送信可能であること。
- `REQ-COM-INT-007` Safety State を送信可能であること。
- `REQ-COM-INT-008` Message の送信元および送信先を識別可能であること。

# 5. 外部通信要求
- `REQ-COM-EXT-001` 外部AIまたは外部計算機へ処理要求を送信可能であること。
- `REQ-COM-EXT-002` 外部AIまたは外部計算機から処理結果を受信可能であること。
- `REQ-COM-EXT-003` ログおよびシステム状態を外部へ送信可能であること。
- `REQ-COM-EXT-004` 必要に応じてモデルまたは設定情報を送受信可能であること。
- `REQ-COM-EXT-005` 外部通信が利用できない場合にローカル処理へ切替可能であること。

# 6. 通信品質要求
- `REQ-COM-QLT-001` Message の欠損または異常を検出可能であること。
- `REQ-COM-QLT-002` 必要な通信について Timeout を設定可能であること。
- `REQ-COM-QLT-003` Message の順序を必要に応じて識別可能であること。
- `REQ-COM-QLT-004` 各Messageに必要に応じて Timestamp を付与可能であること。
- `REQ-COM-QLT-005` 重要度の異なるMessageを区別可能であること。
- `REQ-COM-QLT-006` 高負荷時にも安全関連Messageを優先可能であること。

# 7. 通信断要求
- `REQ-COM-OFF-001` 通信断を上位システムへ通知すること。
- `REQ-COM-OFF-002` 通信断時に危険な指令を保持し続けないこと。
- `REQ-COM-OFF-003` 通信断時に必要なサブシステムへ Degraded 状態を通知可能であること。
- `REQ-COM-OFF-004` 通信復旧後に再接続可能であること。
- `REQ-COM-OFF-005` 再接続時に古い指令を誤って実行しないこと。

# 8. セキュリティ要求
- `REQ-COM-SEC-001` 外部からの不正な指令を無条件に実行しないこと。
- `REQ-COM-SEC-002` 必要に応じて通信相手を識別可能であること。
- `REQ-COM-SEC-003` 制御および安全に関する重要通信を保護可能な構成とすること。

# 9. 拡張性要求
- `REQ-COM-EXTN-001` Ethernet、UART、SPI、I2C、CAN、USB 等の複数通信方式に対応可能な構成とすること。
- `REQ-COM-EXTN-002` 新しい通信方式を追加可能であること。
- `REQ-COM-EXTN-003` Message形式の変更をバージョン管理可能であること。

# 10. 未確定事項
- 通信プロトコル
- Message Format
- Message ID
- 通信周期
- Timeout
- 再送方式
- 暗号化方式
- 認証方式
