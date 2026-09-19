# `include/ai/model/agent_transformer.h`

## 目的

CUDA tensor基盤上の小型decoder-only causal Transformerを定義する。会話、構造化mode、要約などをmode tokenで切り替える共通生成モデルであり、双方向NER encoderではない。

## 構成

`AgentTransformerConfig` は語彙、層数、埋め込み次元、head数、FFN次元、context、dropout、初期化seedを持つ。`validate()` はhead分割や範囲を検査し、`parameter_count()` は構成から学習parameter数を算出する。

各blockはPre-LayerNorm型で、causal multi-head self-attention、残差、GELU FFN、残差を順に適用する。モデル全体はtoken埋め込みと学習可能な位置埋め込みを加え、block列、最終LayerNorm、語彙射影を通す。

## 公開クラス

- `AgentAttention`: Query/Key/Value/出力射影とcausal attention。
- `AgentFeedForward`: 2層の位置別MLP。
- `AgentTransformerBlock`: attentionとFFNを残差接続する単位。
- `AgentTransformer`: token ID入力、語彙logit出力、ignored ID対応lossを提供する。

## 再現性と状態

attention/FFNのdropoutは呼出位置counterを持つ。`dropout_counters()` と `set_dropout_counters()` はcheckpoint再開で乱数系列を復元するためのAPIであり、任意の推論制御用ではない。

## 入力制約

token IDはtokenizer語彙内、系列長はconfigのcontext以下でなければならない。`loss()` の既定ignored IDはpad ID 0で、SFT prompt maskと対応する。

## 主な実装・検証先

- 実装: `src/model/agent_transformer.cpp`
- 学習: [agent_training.md](agent_training.md)
- 生成: [model_language_model.md](model_language_model.md)
- テスト: transformer、attention、checkpoint関連check
