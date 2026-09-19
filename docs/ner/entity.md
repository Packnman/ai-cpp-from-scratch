# `include/ai/ner/entity.h`

## 目的

rule抽出とmodel抽出が共有するentity表現、およびUnicode code point位置と公開UTF-8 byte位置の橋渡しを定義する。

## `EntityType`

先頭8値は上流Stockmark NER annotationと対応する人名、法人名、政治的組織名、その他の組織名、地名、施設名、製品名、イベント名である。続く金額、日付、時刻、期間、数量、条件は会話向けrule抽出型である。先頭8値の順序はmodel bundleのBIO label契約に影響する。

## `EntityMention`

- `start`, `end`: 原文UTF-8のbyte offsetによる半開区間 `[start, end)`。
- `surface`: 必ず `text.substr(start, end-start)` で原文へ戻れる表記。
- `normalized`: 確実に正規化できる場合だけ設定する任意値。
- `source`: rule、model、またはhybrid統合由来。
- `utterance_id`: spanが属する発言の識別子。
- `score`: modelの未校正ranking score。正解確率として扱わない。

`valid_mention()` は範囲とsurfaceの原文一致を検査する。entityが言及されたことと、内容が事実であることは別である。

## UTF-8境界

`decode_utf8()` は入力をUnicode code point列へ変換し、各要素に元byte範囲を保持する。非canonical、途中切れ、surrogate、範囲外code pointを例外にする。modelはcode point単位で推論し、外部APIへ返すときこの対応でbyte spanへ戻す。

## 主な実装・検証先

- 実装: `src/ner/entity.cpp`
- テスト: `tests/ner/ner_check.cpp` およびUTF-8/span関連check
