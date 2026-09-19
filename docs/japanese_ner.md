# 日本語NER

## 実装の位置づけ

NERは生成モデルと独立した `ner_core` と bundle である。公開interfaceは
`include/ai/ner/entity.h` と `include/ai/ner/extractor.h` にあり、
`EntityMention`、`IEntityExtractor`、`RuleEntityExtractor`、
`ModelEntityExtractor`、`HybridEntityExtractor` を提供する。

公開位置は原文UTF-8 byte列の半開区間 `[start,end)` である。学習時だけ
Unicode code point列を使い、`decode_utf8` の各要素がcode point位置とbyte範囲を
対応付ける。全抽出結果は `text.substr(start,end-start)==surface` に戻せる。
モデルの `score` は未校正logitであり、正解確率ではない。

Agentは `--ner off|rule|model|hybrid` で切り替える。既定は `off` なのでbundleを
指定しない既存動作は変わらない。`model` と `hybrid` は `--ner-bundle` が必須で、
実行時の学習やdownloadはない。候補には発話IDと原文位置を保持し、現在入力、
要約payload、memory-write payloadへ `candidate_not_fact` として渡す。候補は
context予算内のoptional sectionで、予算を拡張しない。否定、提案、訂正、現在値の
確定はNERの責務ではない。従って「5万円から3万円に変更」は二つの金額候補を
返すだけで、3万円への状態更新は別のreasoner処理で行う。

## データ、出典、ライセンス

- 上流: Stockmark `ner-wikipedia-dataset` version 2.0
- 固定revision: `5e525c2132c0e87cce890b8b92639a0cf357c1f3`
- `ner.json` SHA-256:
  `5796effb85c473aec5b85784545349ed42e408bb0fecdb78f82bc107d53edbe3`
- 取得URL、revision、checksumは `fetch_ner_wikipedia.sh` と生成される `SOURCE` に
  保存する。
- 上流READMEはWikipedia日本語版と同じCC BY-SA 3.0とし、改変・再配布時には
  Wikipediaの帰属・継承条件を参照するよう求めている。作成者はストックマーク
  株式会社、参考文献は近江崇宏「Wikipediaを用いた日本語の固有表現抽出の
  データセットの構築」（言語処理学会第27回年次大会、2021）。
- 学習データ本文とbundleはGit対象外の `data/` / `models/` に置く。

データのCC BY-SA条件が、個々の学習済み重みへどのように及ぶかは上流READMEに
明記されていない。この実装はデータのライセンス表示をbundleの配布条件と
同一視しない。bundleを第三者へ配布する際の法的評価、必要な帰属文、生成物の
ライセンスは未確認事項である。

## 変換仕様

上流8ラベル（人名、法人名、政治的組織名、その他の組織名、地名、施設名、
製品名、イベント名）をすべて維持する。上流spanはPython Unicode code pointの
半開区間であることを、全entityで `text[start:end]==name` により検証する。
変換後JSONLにはcode point spanとUTF-8 byte spanの両方を保存する。

`curid` をページ単位にし、さらに完全一致文を持つページをunion-findで連結して
から、seed付きSHA-256で80/10/10に分割する。同一ページと完全重複文はsplitを
跨がない。元データには既存splitがない。負例も分割対象に含める。重複・入れ子
spanはBIO不能として件数をmanifestへ記録し、原例を `rejected.jsonl` へ残す。
長文は句読点境界を優先し、entityを跨ぐ境界をentity末尾まで延長する。
testは最終評価だけに使う。

固定データをseed `20260918`、最大256 code pointsで変換した実測は次の通り。

- 元例 5,343、ページ 5,235、負例 484
- train 4,320、validation 526、test 500（長文分割で合計が3増加）
- 長文分割 3、BIO不能 0、完全重複文 0
- 負例はtrain 382、validation 46、test 57

manifestにはラベル別件数、分割seed、fingerprint、変換方針も保存する。

## ルール抽出

金額、絶対日付、時刻、期間、数量、明示的条件節を抽出する。全角数字を扱い、
確実な場合だけ日付をISO形式、時刻を `HH:MM`、金額を通貨付き値へ正規化する。
「明日」などの相対日付は基準日時なしに確定も正規化もしない。単位はsurfaceと
正規化値の双方に残す。同一値の複数出現は別spanになる。

Hybridの重複規則は、数値系の高精度ルールspanを優先し、重なるモデルspanを
除外、重ならない上流8ラベルを保持する。同一層では開始位置順で決定的に返す。

## 学習モデルとbundle

現行tensor層はCUDA生成モデル向けで、CPU recurrent層とautogradを持たない。
Agentに重量級runtime依存を追加せずC++だけで推論するため、初期版はBiLSTMでは
なく、小さい `bidirectional_window` encoderを採用した。trainだけから作る文字
語彙（`<PAD>`, `<UNK>`付き）、文字埋め込み、左・右contextの別集約、tanh層、
17クラスBIO softmaxで構成する。さらにtrainに現れた表記だけの曖昧性解決済み
lexiconをbundleへ持ち、既知表記を優先する。これは双方向contextを使うが、
BiLSTMと同等ではない。

不正な `I-X` は、直前が同じ型でなければ `B-X` として明示的にrepairする。
CRFは未実装。bundle v1は文字語彙、17ラベル対応、encoder設定、重み、lexicon、
データrevision/checksum/split seed、scoreの意味をJSONに保存する。既存の言語
モデルbundleとは別ファイルで互換性を変更しない。

## 実行

一括実行:

```sh
./scripts/run_ner_pipeline.sh
```

個別には `fetch_ner_wikipedia.sh`、`convert_ner_wikipedia.py`、`ner_cli train`、
`ner_cli evaluate`、`ner_cli infer` を使う。例:

```sh
build/ner/ner_cli infer --bundle data/ner/bundle/ner.json --text '田中さんは東京へ行く'
build/ner/agent_cli --ner hybrid --ner-bundle data/ner/bundle/ner.json
```

## 2026-09-18の実測と限界

環境はLinux x86-64、GCC 13.3、Release build、CPU実行。全trainを20 epoch、
embedding 16、hidden 32、window 3、seed 42で学習した。lossは0.8283から
0.4111へ低下し、保存・再ロード・推論を確認した。

validation exact span+type F1は0.1424（P 0.1033 / R 0.2292）。設定確定後に一度
測ったtestはF1 0.1332（P 0.0940 / R 0.2286）だった。testの型別F1は人名
0.0804、法人名0.1162、政治的組織名0.2707、その他の組織名0.1084、地名
0.2752、施設名0.0302、製品名0.0379、イベント名0.1611。十分な精度ではなく、
このbundleを高信頼NERとして採用すべきではない。

重みは175,620 bytes、JSON bundleは1,383,013 bytes。test 28,376 code pointsの
処理は0.435秒（約65,237 code points/秒）、プロセス最大RSSは10,464 KiB。lexiconの線形照合が
速度と偽陽性を悪化させる既知課題である。

ルール単体は、重複金額、全角日付・時刻、句読点、数量、期間、相対日付非確定、
条件節、UTF-8 byte復元を含む28 assertionsを通過した。Wikipedia testだけでは
会話有効性を示さない。Agentの候補伝播とcontext上限は実装・回帰試験済みだが、
同一生成モデルでのNER on/off最終回答精度、反復要約、人名・否定・訂正保持率の
定量比較は、利用可能な学習済み会話bundleがないため未実行である。今後はtestを
触らず、独立した少量会話fixtureで測る必要がある。未知表記では局所window
classifierの境界と型が崩れやすく、特に施設名・製品名・人名が弱い。独立合成例
「架空太郎さんは月面市の新設法人ネコテックへ行く」では、人名を「架」「太」
「郎」へ分断し、月面市を未抽出、ネコテックを部分抽出した。

次の改善候補は、同じinterface/bundle分離を保った文字BiLSTM、class-weighted
学習、遷移制約付きViterbi、lexicon trie化、少量の再現可能な会話評価例である。
