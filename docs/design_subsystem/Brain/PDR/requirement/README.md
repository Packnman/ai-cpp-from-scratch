# 1. 目的

本書は、人型ロボットシステムに搭載する Brain System が満たすべき要求事項を定義する。

Brain System は、外界およびロボット自身の状態を認識し、人間から与えられた指示およびシステム内部で生成された目的を理解し、目的達成に必要な行動を計画・管理することを目的とする。

また、会話、記憶、学習および外部AIとの連携を行い、ロボットが環境および状況に応じた高レベル判断を実行可能とすること。

具体的なニューラルネットワーク構成、アルゴリズム、Token形式、クラス構造、データベース構造等については設計仕様書にて定義する。

# 2. 適用範囲

本書は以下の Brain System 機能を対象とする。

* 外界認識
* 自己状態認識
* 音声認識
* 文脈認識
* 目的認識
* 行動計画
* 計画実行管理
* 記憶
* 学習
* 外部AI連携
* Control Systemとの連携
* Safety Systemとの連携

Brain Systemは高レベル判断を担当し、直接的なモータ電流制御、PWM生成等の低レベル制御は対象外とする。

# 3. 関連文書

| 文書名                       | 内容                         |
| :------------------------ | :------------------------- |
| [システム要求仕様書](../../../design_system/requirement/README.md)                 | ロボットシステム全体の要求              |
| [システム設計仕様書](../../../design_system/spec/README.md) | システム構成およびBrain Systemの位置付け |
| [Brain System設計仕様書](../spec/README.md) | 本要求を実現するためのBrain内部設計 |
| [Control System要求仕様書](../../Control/requirement/README.md) | 動作および制御に関する要求 |
| [Safety System要求仕様書](../../Safety/requirement/README.md) | 安全監視および安全停止に関する要求 |
| [Communication System要求仕様書](../../Communication/requirement/README.md) | 内部・外部通信に関する要求 |

# 4. 前提条件

* `REQ-BRN-PRE-001` Brain SystemはSensor Systemから取得した情報を利用可能であること。
* `REQ-BRN-PRE-002` Brain SystemはControl Systemからロボット状態および行動結果を取得可能であること。
* `REQ-BRN-PRE-003` Brain SystemはSafety Systemから安全状態および制約情報を取得可能であること。
* `REQ-BRN-PRE-004` Brain Systemは必要に応じて外部計算機または外部AIを利用可能であること。
* `REQ-BRN-PRE-005` 外部通信は常時利用可能であることを前提としないこと。
* `REQ-BRN-PRE-006` Sensor入力には誤差、欠損、遅延および誤認識が含まれる可能性を考慮すること。
* `REQ-BRN-PRE-007` Brain Systemの判断結果が常に正しいことを前提としないこと。
* `REQ-BRN-PRE-008` 安全に関する最終的な強制停止および低レベル制限をBrain System単独に依存しないこと。

# 5. Brain System基本要求

* `REQ-BRN-SYS-001` Brain Systemは外界の状態を認識できること。
* `REQ-BRN-SYS-002` Brain Systemはロボット自身の状態を認識できること。
* `REQ-BRN-SYS-003` Brain Systemは人間から与えられた指示を理解できること。
* `REQ-BRN-SYS-004` Brain Systemは達成すべき目的を認識または生成できること。
* `REQ-BRN-SYS-005` Brain Systemは現在状態と目的から行動計画を生成できること。
* `REQ-BRN-SYS-006` Brain Systemは生成した行動計画の実行状態を管理できること。
* `REQ-BRN-SYS-007` Brain Systemは行動結果に応じて計画を変更できること。
* `REQ-BRN-SYS-008` Brain Systemは過去の情報を記憶し、後の判断へ利用できること。
* `REQ-BRN-SYS-009` Brain Systemは新しく得た知識または経験を将来利用可能な形式で保持できること。
* `REQ-BRN-SYS-010` Brain Systemは必要に応じて外部AIまたは外部計算資源を利用できること。
* `REQ-BRN-SYS-011` 外部AIが使用不能な場合でも最低限必要なBrain機能をローカルで継続できること。

# 6. 入力認識要求

## 6.1 外界認識

* `REQ-BRN-PER-001` 画像情報から周囲に存在する物体を認識できること。
* `REQ-BRN-PER-002` 認識した物体の種類を識別できること。
* `REQ-BRN-PER-003` 認識した物体の位置を取得または推定できること。
* `REQ-BRN-PER-004` 周囲の空間構造を認識できること。
* `REQ-BRN-PER-005` 障害物および移動可能領域を認識できること。
* `REQ-BRN-PER-006` 複数の認識情報から同一対象を関連付け可能であること。
* `REQ-BRN-PER-007` 時間的に連続した認識情報から対象の状態変化を認識可能であること。

## 6.2 力・接触認識

* `REQ-BRN-FRC-001` 外力に関する入力情報を認識できること。
* `REQ-BRN-FRC-002` 接触の有無を認識できること。
* `REQ-BRN-FRC-003` 必要に応じて接触位置、力の大きさおよび方向を認識できること。
* `REQ-BRN-FRC-004` 力および接触情報を行動判断へ利用可能であること。

## 6.3 自己状態認識

* `REQ-BRN-STATE-001` ロボット自身の位置および姿勢を認識できること。
* `REQ-BRN-STATE-002` 関節状態を認識できること。
* `REQ-BRN-STATE-003` 現在実行中の行動を認識できること。
* `REQ-BRN-STATE-004` バッテリー、温度、通信等のシステム状態を利用可能であること。
* `REQ-BRN-STATE-005` 異常状態をPlanningで利用可能な状態情報として受け取れること。

# 7. 音声・言語理解要求

## 7.1 音声認識

* `REQ-BRN-SPC-001` 人間の音声入力を認識できること。
* `REQ-BRN-SPC-002` 音声入力を後段の言語処理で利用可能な形式へ変換できること。
* `REQ-BRN-SPC-003` 音声認識結果に認識信頼度を付与可能であること。
* `REQ-BRN-SPC-004` 認識信頼度が不足する場合、確定情報として扱わないこと。

## 7.2 文脈認識

* `REQ-BRN-CTX-001` テキスト入力から発話または文章の意味を認識できること。
* `REQ-BRN-CTX-002` 入力から要求、質問、命令等の意図を識別できること。
* `REQ-BRN-CTX-003` 入力から対象物、場所、人物等を識別できること。
* `REQ-BRN-CTX-004` 入力から条件および制約を抽出可能であること。
* `REQ-BRN-CTX-005` 過去の会話を参照して省略された対象または文脈を補完可能であること。
* `REQ-BRN-CTX-006` 意味を一意に決定できない場合、その不確実性を保持できること。
* `REQ-BRN-CTX-007` 必要に応じて人間へ追加確認を要求できること。

# 8. 共通内部情報要求

Brain Systemは、異なる入力源から得られる情報をPlanningで利用可能な共通概念へ整理できること。

## 8.1 情報分類

少なくとも以下の情報種別を区別して扱えること。

| 種別            | 内容               |
| :------------ | :--------------- |
| Perception    | 外界から認識した情報       |
| Robot State   | ロボット自身の現在状態      |
| Goal          | 達成すべき目的          |
| Condition     | 現在成立している状態または条件  |
| Constraint    | 行動時に守るべき制約       |
| Memory        | 過去から取得した情報       |
| Policy        | 行動方法または判断に利用する方策 |
| Action Result | 実行した行動の結果        |

* `REQ-BRN-INFO-001` 外界認識結果をPerceptionとして表現可能であること。
* `REQ-BRN-INFO-002` ロボット自身の状態をRobot Stateとして表現可能であること。
* `REQ-BRN-INFO-003` 達成すべき目的をGoalとして表現可能であること。
* `REQ-BRN-INFO-004` 現在成立している前提または状態をConditionとして表現可能であること。
* `REQ-BRN-INFO-005` 行動時に守るべき条件をConstraintとして表現可能であること。
* `REQ-BRN-INFO-006` 過去情報をMemoryとしてPlanningから利用可能であること。
* `REQ-BRN-INFO-007` 学習済みまたは事前定義された行動方法をPolicyとして利用可能であること。
* `REQ-BRN-INFO-008` Executionから得られた結果をAction Resultとして利用可能であること。

## 8.2 共通属性

内部情報には必要に応じて以下を付与可能とする。

| 属性         | 内容              |
| :--------- | :-------------- |
| ID         | 情報を識別するための識別子   |
| Type       | 情報の種類           |
| Value      | 情報内容            |
| Relation   | 他情報との関係         |
| Confidence | 情報の信頼度          |
| Timestamp  | 情報が観測または生成された時刻 |
| Source     | 情報の取得元          |

* `REQ-BRN-INFO-009` 内部情報を一意または必要な範囲で識別可能であること。
* `REQ-BRN-INFO-010` 情報間の関連を保持可能であること。
* `REQ-BRN-INFO-011` 不確実な情報についてConfidenceを保持可能であること。
* `REQ-BRN-INFO-012` 時間依存情報についてTimestampを保持可能であること。
* `REQ-BRN-INFO-013` 情報の取得元または生成元を識別可能であること。
* `REQ-BRN-INFO-014` 情報種別固有の属性を追加可能であること。

例としてPositionは物体および空間情報には必要となるが、GoalやConstraint等では必須としない。

# 9. Goal要求

* `REQ-BRN-GOAL-001` Brain Systemは達成すべき目的をGoalとして保持できること。
* `REQ-BRN-GOAL-002` 人間からの指示をGoalへ変換可能であること。
* `REQ-BRN-GOAL-003` ロボット自身の状態から必要に応じてGoalを生成可能であること。
* `REQ-BRN-GOAL-004` Goalは対象を保持可能であること。
* `REQ-BRN-GOAL-005` Goalは達成条件を保持可能であること。
* `REQ-BRN-GOAL-006` 複数のGoalが存在する場合、優先関係を扱えること。
* `REQ-BRN-GOAL-007` Goal達成、取消し、失敗等の状態を管理できること。

# 10. Condition要求

* `REQ-BRN-CND-001` 現在成立している環境状態をConditionとして保持可能であること。
* `REQ-BRN-CND-002` 現在成立しているRobot StateをConditionとして利用可能であること。
* `REQ-BRN-CND-003` PlanningはAction実行に必要なConditionを評価可能であること。
* `REQ-BRN-CND-004` Conditionが成立しないActionを無条件に実行しないこと。
* `REQ-BRN-CND-005` 状態変化によってConditionが変化した場合、Planningへ反映可能であること。

# 11. Constraint要求

* `REQ-BRN-CST-001` Brain Systemは行動に対する制約を保持可能であること。
* `REQ-BRN-CST-002` Constraintは必須制約と許容可能な制約を区別可能であること。
* `REQ-BRN-CST-003` 必須制約に違反するAction Planを実行対象としないこと。
* `REQ-BRN-CST-004` Safety Systemから入力された安全制約をPlanningへ反映すること。
* `REQ-BRN-CST-005` 人間から指定された条件をConstraintとして扱えること。
* `REQ-BRN-CST-006` 状況変化により新たなConstraintが発生した場合、既存Planを再評価できること。

# 12. Planning要求

## 12.1 基本Planning

* `REQ-BRN-PLN-001` Goal達成に必要なAction Planを生成できること。
* `REQ-BRN-PLN-002` PerceptionをPlanningに利用できること。
* `REQ-BRN-PLN-003` Robot StateをPlanningに利用できること。
* `REQ-BRN-PLN-004` GoalをPlanningの目的として利用できること。
* `REQ-BRN-PLN-005` ConditionをAction実行可否判断へ利用できること。
* `REQ-BRN-PLN-006` ConstraintをPlanningへ反映できること。
* `REQ-BRN-PLN-007` MemoryをPlanningへ利用できること。
* `REQ-BRN-PLN-008` PolicyをPlanningへ利用できること。
* `REQ-BRN-PLN-009` Action Resultを後続Planningへ利用できること。

## 12.2 Action Plan

* `REQ-BRN-PLN-010` Action Planは1個以上のActionから構成可能であること。
* `REQ-BRN-PLN-011` Action間の実行順序を定義可能であること。
* `REQ-BRN-PLN-012` 必要に応じてActionの並列実行関係を定義可能であること。
* `REQ-BRN-PLN-013` Actionの対象を指定可能であること。
* `REQ-BRN-PLN-014` Actionの完了条件を指定可能であること。
* `REQ-BRN-PLN-015` Actionの失敗条件を扱えること。

## 12.3 再計画

* `REQ-BRN-REPLN-001` 対象状態が変化した場合に再計画可能であること。
* `REQ-BRN-REPLN-002` Actionが失敗した場合に再計画可能であること。
* `REQ-BRN-REPLN-003` Goalが変更された場合に再計画可能であること。
* `REQ-BRN-REPLN-004` Constraintが変更された場合に再計画可能であること。
* `REQ-BRN-REPLN-005` Action実行条件が成立しなくなった場合に再計画可能であること。
* `REQ-BRN-REPLN-006` Safety Systemから制限要求を受信した場合、必要に応じてPlanを変更または中断できること。

# 13. Execution管理要求

Brain SystemにおけるExecutionは、低レベル制御そのものではなく、Action Planの実行管理を対象とする。

* `REQ-BRN-EXEC-001` Action Planを順次実行管理できること。
* `REQ-BRN-EXEC-002` ActionをControl Systemへ指示可能であること。
* `REQ-BRN-EXEC-003` Actionの実行開始を管理できること。
* `REQ-BRN-EXEC-004` Actionの実行中状態を管理できること。
* `REQ-BRN-EXEC-005` Actionの成功を認識できること。
* `REQ-BRN-EXEC-006` Actionの失敗を認識できること。
* `REQ-BRN-EXEC-007` ActionのTimeoutを扱えること。
* `REQ-BRN-EXEC-008` Actionを中断可能であること。
* `REQ-BRN-EXEC-009` 実行結果をAction ResultとしてPlanningへ返却できること。
* `REQ-BRN-EXEC-010` 実行結果を記憶処理へ渡すことができること。

# 14. 会話要求

* `REQ-BRN-CNV-001` 日常会話を処理可能であること。
* `REQ-BRN-CNV-002` 特定分野における専門的な会話を処理可能であること。
* `REQ-BRN-CNV-003` 会話履歴を考慮した応答が可能であること。
* `REQ-BRN-CNV-004` 登録済みの専門用語および固有語句を利用可能であること。
* `REQ-BRN-CNV-005` 質問に対して必要な記憶または知識を検索可能であること。
* `REQ-BRN-CNV-006` 回答に必要な情報が不足する場合、不足を認識可能であること。
* `REQ-BRN-CNV-007` 不明な内容について事実として断定しないこと。

# 15. Memory要求

## 15.1 Short-Term Memory

* `REQ-BRN-STM-001` 現在のGoalを保持可能であること。
* `REQ-BRN-STM-002` 現在のAction Planを保持可能であること。
* `REQ-BRN-STM-003` 最近の認識結果を保持可能であること。
* `REQ-BRN-STM-004` 最近の会話を保持可能であること。
* `REQ-BRN-STM-005` 最近のAction Resultを保持可能であること。
* `REQ-BRN-STM-006` 不要となった情報を削除または圧縮可能であること。

## 15.2 Long-Term Memory

* `REQ-BRN-LTM-001` 長期利用価値のある情報を永続的に保存可能であること。
* `REQ-BRN-LTM-002` 人物に関する情報を保存可能であること。
* `REQ-BRN-LTM-003` 物体に関する情報を保存可能であること。
* `REQ-BRN-LTM-004` 場所に関する情報を保存可能であること。
* `REQ-BRN-LTM-005` 語句および専門知識を保存可能であること。
* `REQ-BRN-LTM-006` 過去の行動およびその結果を保存可能であること。
* `REQ-BRN-LTM-007` 保存情報を後のPlanningまたは会話から検索可能であること。
* `REQ-BRN-LTM-008` 記憶情報を更新可能であること。
* `REQ-BRN-LTM-009` 不要または陳腐化した記憶を整理可能であること。

# 16. 学習要求

* `REQ-BRN-LRN-001` 新しい語句を習得可能であること。
* `REQ-BRN-LRN-002` 新しい物体または物体名称を登録可能であること。
* `REQ-BRN-LRN-003` 新しい場所を登録可能であること。
* `REQ-BRN-LRN-004` Actionの成功および失敗を経験情報として保存可能であること。
* `REQ-BRN-LRN-005` 保存した経験情報を将来の判断改善へ利用可能であること。
* `REQ-BRN-LRN-006` 学習済み情報の出所または生成経緯を必要に応じて識別可能であること。
* `REQ-BRN-LRN-007` 学習による変更が安全制約を無効化しないこと。

Brain Systemのオンライン動作中にモデル重みを直接更新するか、学習データのみ蓄積して後から再学習するかは設計仕様書にて定義する。

# 17. 外部AI連携要求

システム設計では、高負荷認識、Large Modelおよび重いPlanningを外部計算資源へ分散可能な構成としている。

* `REQ-BRN-EXT-001` Brain Systemは外部AIへ処理を要求可能であること。
* `REQ-BRN-EXT-002` 外部AIから処理結果を取得可能であること。
* `REQ-BRN-EXT-003` 外部AI処理中もローカルSafety機能を妨げないこと。
* `REQ-BRN-EXT-004` 外部AIからの応答が得られない場合を検出可能であること。
* `REQ-BRN-EXT-005` 外部AIの応答がTimeoutした場合、処理を継続待機し続けないこと。
* `REQ-BRN-EXT-006` 外部AIから得られた情報を無条件に安全な情報として扱わないこと。
* `REQ-BRN-EXT-007` 外部AI切断時にローカル動作へ移行可能であること。
* `REQ-BRN-EXT-008` 通信復旧後に外部AI連携へ復帰可能であること。

# 18. 通信断時要求

システム全体として通信断時にも基本認識、基本行動、状態監視等を継続する構成が定義されている。

* `REQ-BRN-OFF-001` 外部通信切断を検出可能であること。
* `REQ-BRN-OFF-002` 通信断時に外部AIを必要とする処理を停止または代替可能であること。
* `REQ-BRN-OFF-003` 通信断時でも最低限の外界認識を継続可能であること。
* `REQ-BRN-OFF-004` 通信断時でも最低限の自己状態認識を継続可能であること。
* `REQ-BRN-OFF-005` 通信断時でも最低限のローカルPlanningが可能であること。
* `REQ-BRN-OFF-006` 通信断時でもControl Systemへの基本行動指示が可能であること。
* `REQ-BRN-OFF-007` 通信断によって安全機能を喪失しないこと。
* `REQ-BRN-OFF-008` 通信状態をPlanningにおけるConditionとして利用可能であること。

# 19. Safety連携要求

* `REQ-BRN-SAFE-001` Brain SystemはSafety Systemの状態を取得可能であること。
* `REQ-BRN-SAFE-002` Safety Systemから指定された制約をPlanningに反映すること。
* `REQ-BRN-SAFE-003` Safety Systemによって禁止されたActionを実行要求しないこと。
* `REQ-BRN-SAFE-004` Safety SystemによるAction中断要求を受け付け可能であること。
* `REQ-BRN-SAFE-005` Emergency Stop状態では新規Actionを開始しないこと。
* `REQ-BRN-SAFE-006` 安全状態からの復帰条件を満たさない状態で自動的に通常動作へ復帰しないこと。
* `REQ-BRN-SAFE-007` Brain System自身の異常時にControl Systemを危険な状態へ遷移させないこと。

Safety System自体はBrainから独立し、BrainまたはAIの異常指令を制限・停止できる構成とする。

# 20. Control Systemインタフェース要求

* `REQ-BRN-CTRL-001` Brain SystemはControl SystemへAction Commandを出力可能であること。
* `REQ-BRN-CTRL-002` Action CommandはControl Systemが解釈可能な形式であること。
* `REQ-BRN-CTRL-003` Brain SystemはControl SystemからAction Resultを取得可能であること。
* `REQ-BRN-CTRL-004` Control Systemから現在状態を取得可能であること。
* `REQ-BRN-CTRL-005` Action CommandとAction Resultを対応付け可能であること。
* `REQ-BRN-CTRL-006` Control Systemの異常をPlanningへ反映可能であること。

# 21. 処理性能要求

* `REQ-BRN-PERF-001` Brain Systemはロボットの動作に必要な時間内に認識結果を生成できること。
* `REQ-BRN-PERF-002` Brain Systemは実用上許容可能な時間内にAction Planを生成できること。
* `REQ-BRN-PERF-003` Brain Systemは会話に必要な時間内に応答を生成できること。
* `REQ-BRN-PERF-004` 高負荷処理によってSafety関連情報の処理が停止しないこと。
* `REQ-BRN-PERF-005` CPU、GPU/NPU、Memory等のリソース不足を検出可能であること。
* `REQ-BRN-PERF-006` リソース不足時に重要度の低い処理を制限可能であること。

具体的な処理周期、最大応答時間および使用可能リソース量は設計または試作評価によって定める。

# 22. エラー処理要求

* `REQ-BRN-ERR-001` Brain内部で発生したエラーを検出可能であること。
* `REQ-BRN-ERR-002` エラーの発生元を識別可能であること。
* `REQ-BRN-ERR-003` エラー発生時刻を記録可能であること。
* `REQ-BRN-ERR-004` 回復可能なエラーについて再試行可能であること。
* `REQ-BRN-ERR-005` 機能停止時に代替処理へ切替可能であること。
* `REQ-BRN-ERR-006` 回復不能な場合、安全側へ遷移可能であること。
* `REQ-BRN-ERR-007` エラー発生時に原因調査可能な情報を残すこと。

# 23. ログ要求

* `REQ-BRN-LOG-001` 認識結果を記録可能であること。
* `REQ-BRN-LOG-002` Goalを記録可能であること。
* `REQ-BRN-LOG-003` Conditionを記録可能であること。
* `REQ-BRN-LOG-004` Constraintを記録可能であること。
* `REQ-BRN-LOG-005` Action Planを記録可能であること。
* `REQ-BRN-LOG-006` Action Resultを記録可能であること。
* `REQ-BRN-LOG-007` Memory更新内容を必要に応じて記録可能であること。
* `REQ-BRN-LOG-008` 外部AIとの通信状態を記録可能であること。
* `REQ-BRN-LOG-009` 使用したAIモデルのバージョンを識別可能であること。
* `REQ-BRN-LOG-010` 各ログにTimestampを付与可能であること。

# 24. 拡張性要求

* `REQ-BRN-EXTN-001` 新しい認識機能を追加可能であること。
* `REQ-BRN-EXTN-002` 新しい入力種別を追加可能であること。
* `REQ-BRN-EXTN-003` 新しいActionを追加可能であること。
* `REQ-BRN-EXTN-004` 新しいPolicyを追加可能であること。
* `REQ-BRN-EXTN-005` AIモデルを交換可能な構成であること。
* `REQ-BRN-EXTN-006` 記憶方式を変更または拡張可能な構成であること。
* `REQ-BRN-EXTN-007` 外部AIの実装変更によるBrain System全体への影響を最小限とすること。
* `REQ-BRN-EXTN-008` 計算装置変更による上位ロジックへの影響を最小限とすること。

# 25. 試験性要求

* `REQ-BRN-TST-001` 各認識機能を個別に試験可能であること。
* `REQ-BRN-TST-002` Planning機能を実機なしで試験可能であること。
* `REQ-BRN-TST-003` Sensor入力を模擬可能であること。
* `REQ-BRN-TST-004` Control SystemからのAction Resultを模擬可能であること。
* `REQ-BRN-TST-005` 外部AI接続状態を模擬可能であること。
* `REQ-BRN-TST-006` 通信断状態を模擬可能であること。
* `REQ-BRN-TST-007` 異常状態およびSafety Constraintを模擬可能であること。
* `REQ-BRN-TST-008` 同一入力条件で問題を再現可能な情報を記録できること。

# 26. 品質要求

* `REQ-BRN-QUAL-001` Brain Systemの一部機能停止によってシステム全体が即座に危険状態へ遷移しないこと。
* `REQ-BRN-QUAL-002` モジュール単位で機能を交換可能な構成とすること。
* `REQ-BRN-QUAL-003` 使用するAIモデルをバージョン管理可能であること。
* `REQ-BRN-QUAL-004` 設定値をバージョン管理可能であること。
* `REQ-BRN-QUAL-005` Memoryのデータ構造をバージョン管理可能であること。
* `REQ-BRN-QUAL-006` 内部情報形式の変更を管理可能であること。
* `REQ-BRN-QUAL-007` ソフトウェア更新後に回帰試験を実施可能であること。

# 27. 要求トレーサビリティ

Brain Systemはシステム設計上、認識、会話、行動計画、記憶、学習および外部AI連携を担当する。

| システム要求        | Brain要求                                         |
| :------------ | :---------------------------------------------- |
| REQ-SYS-PER-* | REQ-BRN-PER-* / REQ-BRN-FRC-* / REQ-BRN-STATE-* |
| REQ-SYS-COM-* | REQ-BRN-SPC-* / REQ-BRN-CTX-* / REQ-BRN-CNV-*   |
| REQ-SYS-ACT-* | REQ-BRN-GOAL-* / REQ-BRN-PLN-* / REQ-BRN-EXEC-* |
| REQ-SYS-MEM-* | REQ-BRN-STM-* / REQ-BRN-LTM-* / REQ-BRN-LRN-*   |
| REQ-SYS-AUT-* | REQ-BRN-EXT-* / REQ-BRN-OFF-*                   |
| REQ-SAFE-*    | REQ-BRN-CST-* / REQ-BRN-SAFE-*                  |
| REQ-QUAL-*    | REQ-BRN-LOG-* / REQ-BRN-TST-* / REQ-BRN-QUAL-*  |
