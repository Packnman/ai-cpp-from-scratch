# `include/ai/ner/extractor.h`

## 目的

NER利用側を抽出方式から分離する `IEntityExtractor` と、rule、学習済みmodel、両者の統合実装を提供する。全実装は原文と任意の発言IDを受け、`EntityMention` 列を返す。

## `RuleEntityExtractor`

金額、完全または部分日付、時刻、期間、数量、明示的な条件markerを決定的に抽出する。全角数字・全角小数記号を認識する。金額、完全年月日、時刻など確実な場合のみ正規化し、相対日付や文脈を要する部分日付は確定しない。

patternは意味的に具体的な順に適用し、既採用の数値spanと重なる後続候補は採用しない。条件spanは文境界までで、数値entityと共存できる。入力UTF-8と全出力spanを検証する。

## `ModelEntityExtractor`

`ai_cpp_ner_bundle_v1` を明示パスからloadするmove-only実装である。runtimeで学習やdownloadは行わない。文字をUnicode code pointとして扱い、未知文字IDを持つ双方向window encoderで8種類のBIO labelを予測する。

`I-X` の前に同種の有効spanがない場合は `B-X` として修復する。言及scoreはspan内logitの最小値で未校正である。bundle内lexiconは長い表記を優先し、重なるneural spanより優先される。`parameter_bytes()` はfloat重みのbyte数で、object/container overheadは含まない。

## `HybridEntityExtractor`

ruleとmodelの両方を必須とする。高精度な数値系rule spanと重なるmodel spanは捨て、非重複のmodel spanを `Hybrid` sourceとして追加する。条件spanとの重なりだけではmodel spanを排除しない。最終結果は開始位置、終了位置順に決定的に並べる。

## 責務境界

抽出結果は記憶・議論状態への「根拠付き候補」であり、無条件の確定事実ではない。属性対応、否定、訂正、撤回、現行値への反映はAgent側で検証して行う。

## 主な実装・検証先

- rule/hybrid: `src/ner/rule_extractor.cpp`
- model: `src/ner/model_extractor.cpp`
- テスト: `tests/ner/ner_check.cpp`、bundle/Agent統合関連check
