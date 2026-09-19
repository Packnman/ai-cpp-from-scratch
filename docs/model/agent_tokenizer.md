# `include/ai/model/agent_tokenizer.h`

## 目的

日本語本文とAgent制御modeを同一token空間へ写すSentencePiece wrapperである。モデルと必ず対で保存し、token IDの意味がずれないようfingerprintを提供する。

## ID契約

`0..3` は `<pad>`, `<unk>`, `<s>`, `</s>`、`4..12` は `AgentMode` と同順の `<CHAT>` から `<TOOL>` である。`special_count` は13。これらの数値はbundle・学習データとの互換性境界なので、既存bundleを移行せず変更してはならない。

## 学習経路

- `train()`: 単一JSONLの `text` または `utterances[].text` から学習する。
- `train_balanced()`: Wikipediaと会話の採用byte数を個別に制限し、一方の規模だけで語彙が決まることを抑える。
- byte fallbackにより未登録文字も表現可能にする。UTF-8文字の途中では学習入力を分断しない。
- `training_metadata()` は抽出元と再現条件を記録するJSONであり、学習済みpiece列とは別の監査情報である。

## 推論・評価

`encode()`/`decode()` は通常文字列だけを扱い、BOS/EOS/mode tokenの付加は呼び出し側の責務である。`evaluate_tokenizer()` は文書数、byte数、文字数、token数、byte fallback数、context超過数を返す。

## 所有権と永続化

SentencePiece processorは共有された不変 `Holder` に隠蔽する。`save()`/`load()` はserialized modelとmetadataを保持し、`fingerprint()` で内容同一性を確認できる。

## 主な実装・検証先

- 実装: `src/model/agent_tokenizer.cpp`
- テスト: `tests/model/agent_model_check.cpp` のtokenizer・効率評価関連case
