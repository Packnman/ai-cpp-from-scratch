# 日本語本文追加・モデル大型化の比較基盤

更新日: 2026-09-13 UTC。

## 実装済み範囲と実測状態

A/B/C/Dを同じtokenizer、最適化token予算、実効batch token、固定質問、response
validation loss/perplexityで比較する実行経路を実装した。リポジトリ内には固定済みjawiki
dumpまたは抽出本文が存在しないため、30,000,000 tokenの本比較は未実行であり、比較値を
記載していない。scripts/run_quality_comparison.sh が実行後に
experiments/quality-comparison/summary.md を生成する。これは未実測値を推測で埋めない
ための明示的な状態である。

専用fixtureでは、本文JSONLの次token予測、外部BPEの不変保存、token予算停止、勾配累積、
metrics/checkpointへの実最適化token数保存を確認した。RTX 3060 Ti 8GBで大型構成を
micro-batch 4、context 512で1 batch実測し、デバイス使用量サンプルは4,218,945,536
bytes、pool予約量は3,087,007,744 bytesだった。既存の現行構成測定ではmicro-batch 16が
成功済みである。したがって既定値は次のとおりで、どちらも1 updateあたり名目8192 token
にそろう。

| 構成 | layers | embedding | heads | FFN | context | micro-batch | accumulation | 実効token/update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 現行 | 4 | 256 | 4 | 1024 | 512 | 16 | 1 | 8192 |
| 大型 | 6 | 384 | 6 | 1536 | 512 | 4 | 4 | 8192 |

## jawikiを固定して準備する

[公式dump](https://dumps.wikimedia.org/jawiki/latest/)で利用する日付を決めた後、
latestではなく日付入りURLを記録し、dumpのSHA-256を別経路でも確認する。
WikiExtractorもcommitを固定し、main namespaceだけをJSON出力する。例:

~~~sh
sha256sum jawiki-YYYYMMDD-pages-articles-multistream.xml.bz2

python WikiExtractor.py --json --processes 8 --namespaces 0 \
  --output extracted-jawiki \
  jawiki-YYYYMMDD-pages-articles-multistream.xml.bz2

scripts/prepare_jawiki.py extracted-jawiki data/jawiki-YYYYMMDD \
  --dump-file jawiki-YYYYMMDD-pages-articles-multistream.xml.bz2 \
  --dump-date YYYYMMDD \
  --dump-url https://dumps.wikimedia.org/jawiki/YYYYMMDD/jawiki-YYYYMMDD-pages-articles-multistream.xml.bz2 \
  --dump-sha256 ACTUAL_SHA256 \
  --extractor-revision WIKIEXTRACTOR_COMMIT
~~~

準備スクリプトは元dumpのSHA-256を検証し、抽出器revision、段落条件、split規則、件数、
dump URL、[Wikimedia利用規約](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en)
をmetadata.jsonへ保存する。page IDのhashで90/5/5へ分けるため、段落がsplitをまたがない。
再配布時は元記事の帰属情報を保持し、学習済み重み公開前にも適用ライセンスを確認する。

## 共通tokenizerと学習経路

共通BPEは会話trainとjawiki trainだけから一度作る。validation/testは渡さない。

~~~sh
build/quality/main_train tokenizer \
  data/conversation/train.jsonl data/jawiki-YYYYMMDD/train.jsonl \
  experiments/quality/common-tokenizer.model --vocab-size 4096
~~~

新規学習では --tokenizer-model FILE でこのモデルを読み、bundleへ同じbytesを保存する。
本文JSONLは {"id":整数,"text":"本文"} 形式で、--data-format text が話者tokenを挟まず
BEGIN, 本文, END の次token予測を行う。会話は --data-format conversation
--loss-target response を使う。--token-budget はtrainで実際に損失対象となったtokenを
数え、到達batchで停止するため、端数batch分だけ超過し得る。実数は各train metricの
optimized_tokens とcheckpointへ保存される。--accumulate はmicro-batch間で勾配を
平均してからclipとAdam updateを行う。

## 比較手順

~~~sh
TEXT_DATA=data/jawiki-YYYYMMDD \
OUTPUT_ROOT=experiments/quality-comparison \
BUILD_DIR=build/quality-comparison \
scripts/run_quality_comparison.sh
~~~

ランナーのtoken配分は次のとおり。

| ID | モデル | 本文次token | 会話response | 合計 |
|---|---|---:|---:|---:|
| A | 現行 | 0 | 30,000,000 | 30,000,000 |
| B | 現行 | 15,000,000 | 15,000,000 | 30,000,000 |
| C | 大型 | 0 | 30,000,000 | 30,000,000 |
| D | 大型 | 15,000,000 | 15,000,000 | 30,000,000 |

通常実行はA/B/Cまでである。AよりBとCの両方が固定5問とresponse validation指標で改善した
と判断した後だけ、RUN_D=1でDを実行する。既存出力ディレクトリは上書きしない。
各完成モデルは同じ固定5問をtop-k 1とtop-k 40で生成し、response対象validation
loss/perplexityも保存する。A対Bをデータ効果、A対Cをモデル規模効果として判定する。
summary.mdの数値だけでなく、各fixed-chatログの回答、END位置、reset通知も確認する。
