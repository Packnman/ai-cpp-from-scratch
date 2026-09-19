# 根拠付き比較・判断: 実装計画と設計

## 現状調査（2026-09-18）

リポジトリ内に `AGENTS.md` は存在しない。作業開始時点の未コミット変更は
会話要約、学習データ、NERに及んでいるため、それらを前提として差分を追加する。

- `DefaultPlanner` は `IReasoner::plan` の結果を渡す。Rule backendは質問や複雑な
  要求を `TaskType::Reasoning` にする。
- `DefaultExecutor` はToolならregistryを呼ぶ一方、Reasoningでは実処理をせず
  operation文字列を `{"result":...}` に入れて成功としていた。
- `DefaultEvaluator` はReasonerへ状態評価を委譲する。Rule backendはToolStatusだけを
  見るため、上記の空のReasoningも成功になる。
- `DefaultAggregator` は別結果に同じJSONキーで違うprimitive値があるだけで矛盾と
  判定していた。対象、時点、条件、訂正を見ないため意味的矛盾ではない。
- Agentは実行を最大2 retry、最大3 replanに制限する。計画は最大32 taskである。
- contextは1024 token固定で、BOSとmode tokenに2、Evaluate/MemoryQuery出力に128、
  その他の出力に256を予約する。入力上限はそれぞれ894または766である。
- NER候補はUTF-8 byte spanと発話IDを持ち、現在入力、要約payload、memory-writeへ
  未確定候補として渡る。要約は6 sectionの検証済み形式で、失敗時は以前の構造化
  要約と履歴を保持する。SQLite memory schemaはversion 1で、本文中心の保存である。
- SFT生成はscenarioではなく各modeの合成例を出し、seed、split、provenance、
  fingerprintを記録する。既存bundleのmode IDを変更できない。

## 実装順

1. scenario単位で分割する再現可能な限定比較データと、現行相当baselineを含む
   評価器を作る。
2. `DiscussionState`、`Claim`、`Evidence`、`Decision`、`StateUpdate` を追加する。
3. 数値、単位、上限・下限・等値、欠損、同一subject/time/conditionの衝突をC++で
   決定的に検証する。
4. `IReasoningTaskExecutor` と限定操作 `discussion.compare` をExecutorへ接続する。
   未知Reasoning operationは成功にしない。
5. 訂正を検証後かつupdate ID単位で冪等適用し、旧値と根拠をrevisionとして残す。
   AgentStateへ選択済み議論状態を返す。要約失敗はこの状態を変更しない。
6. scenarioから正解構造化出力を生成し、既存SFTへ追加可能なJSONLを作る。
   新modeは設けず、既存bundleがこの教師を学習済みだとは扱わない。
7. 抽出、更新、比較、Agent統合、通常会話・tool回帰を別々にテストし、精度、根拠、
   hallucination、訂正、誤矛盾、時間、RSS、JSONサイズを報告する。

## 初期スコープと入力契約

初期対象は `/compare JSON` で与えた資料内の候補、数値fact、必須constraint、
priority、訂正だけである。Web検索、自由討論、一般自然文の真偽判定、複数Agent、
長い思考過程は対象外とする。

資料のfactは `subject/attribute/time/condition/value/unit/evidence_ids/stance` を持つ。
Evidenceは `origin` を `user/document/tool/model/verified_judgement` に分ける。
出典がdocumentにあることは「その資料がそう記述した」根拠であり、世界で真である
保証ではない。未検証のuser/model主張だけでは選択に使わない。constraintは利用者の
選好なので、user由来でも適用できる。

判断statusは `supported`、`insufficient_evidence`、`conflicting_evidence`、
`no_feasible_option` の4種とする。出力には短いconclusion、既知のevidence IDだけ、
open questions、選択肢、選択に使った決定的検証結果を含める。形式検証と判断の
正しさは別指標で測る。

訂正は `correct_constraint`、`correct_claim`、`set_priorities`、`retract_claim` に限定する。適用済み
update IDは再適用せず、置換前をrevision logに保持する。NERは属性対応の候補として
残るが、この初期JSON契約では属性と値の対応を明示入力する。NER抽出だけでClaimを
verifiedへ昇格しない。

## contextと永続化

決定的比較はモデルcontextを消費しない。モデルが最終文を生成する場合も、渡すのは
現在のdecision、使用evidence、open questions、直近revisionだけである。必須情報が
JSON入力上限を超える場合は `insufficient_evidence` とし、黙って削らない。

DiscussionStateは初期実装ではAgent process内メモリにのみ保持し、SQLite schemaを
変更しない。従って既存DBと完全互換である。永続化は未実装であり、将来行う場合は
別tableとschema version migrationが必要になる。

## 完了判定

決定的fixtureでstatus、選択、evidence参照、訂正、誤矛盾を測る。モデル文章がJSONを
出せただけでは成功にしない。会話モデルの追加SFTはsmoke/fullを分け、実行しなければ
未実行と明記する。既存bundleは追加教師を学習していないので対応済みとは表示しない。
