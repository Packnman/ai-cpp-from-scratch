# `include/ai/model/model_language_model.h`

## 目的

学習済みTransformerとtokenizerを一体のbundleとして保存・読込し、`ai::agent::ILanguageModel` としてAgent runtimeへ接続する。

## 生成設定

`GenerationConfig` はtemperature、top-p、seed、最大生成token数、greedy指定を持つ。`validate()` が範囲を検査する。`generate()` は渡されたprompt token列の末尾から自己回帰生成し、EOS、最大生成数、またはmodel contextで停止する。

## bundle契約

`AgentBundle` は `AgentTransformer` と `AgentTokenizer` の所有権をまとめる。`save_bundle()`/`load_bundle()` は構成、重み、tokenizer、manifest上のfingerprintを対応付ける。NER bundleは別形式・別ライフサイクルであり、ここには含まれない。

## Agent adapter

`ModelLanguageModel::complete()` は `ModelMode` を固定IDの `AgentMode` に変換し、BOSとmode tokenを付加して生成する。構造処理modeでは再現性を優先した生成設定を使い、通常応答系のsampling seedはrequest counterでずらす。`token_count()` はcontext構築側の予算計算に使われる。

## 互換性と制約

既存のmode ID、語彙fingerprint、モデル構成を黙って読み替えない。runtimeで学習やbundleの自動取得は行わない。最大token指定は予約量であり、入力と合わせてmodel contextを超えられない。

## 主な実装・検証先

- 実装: `src/model/model_language_model.cpp`
- Agent interface: `include/ai/agent/components.h`
- テスト: bundle round-trip、generation、conversation runtime関連check
