# 日本語本文追加・モデル大型化の比較基盤

更新日: 2026-09-13 UTC。

## 実装済み範囲と実測状態

A/B/C/Dを同じtokenizer、最適化token予算、実効batch token、固定質問、response
validation loss/perplexityで比較する実行経路を実装した。2026-09-01版jawikiコーパスは
`data/jawiki-20260901/` に準備済みである。A/B/C学習はコーパス準備作業には含めて
いないため、30,000,000 tokenの本比較は未実行であり、比較値はまだ記載していない。
`scripts/run_quality_comparison.sh` が実行後に
`experiments/quality-comparison/summary.md` を生成する。これは未実測値を推測で埋めない
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

## 固定済みjawikiコーパス

2026-09-01版の
[公式dump](https://dumps.wikimedia.org/jawiki/20260901/dumpstatus.json)が完了済みであることを
確認し、日付入りURLから取得した。元dumpと公式状態ファイルは
`data/raw/jawiki-20260901/`、WikiExtractorの中間JSONは
`data/intermediate/jawiki-20260901/` に保持している。

| 項目 | 実測値 |
|---|---:|
| 元dump容量 | 4,852,895,748 bytes |
| 公式MD5 | `be3d5a8c1c7a1804628c1162f0a708df` |
| 公式SHA-1 | `98be46ea8ba77352ece82a0b2c00863ddebc1c36` |
| ローカルSHA-256 | `888a2a81e7e2888f364953a69d5fc55b88449df8977eb2be01316a4c53da7252` |
| WikiExtractor revision | `e96bee708b6a0f3ee66a4b677294faa80e3557a6` |
| WikiExtractor条件 | Python 3、8 process、namespace 0、JSON、template展開あり |
| 抽出記事数 | 1,516,319 |
| 中間抽出物 | 10,086 files、10,312,557,505 bytes |

最終コーパスは空行区切りの段落を単位とし、空白を正規化して100文字未満を除外した。
件数制限は設けていない。page IDのSHA-256先頭byteを20で割った剰余により、同じpageの
段落がsplitをまたがない決定的90/5/5分割にした。

| split | 文書数 | JSONL容量 | 本文文字数 | 比率 |
|---|---:|---:|---:|---:|
| train | 1,273,561 | 4,830,237,596 bytes | 1,685,610,310 | 90.63% |
| validation | 65,572 | 246,544,279 bytes | 86,139,219 | 4.67% |
| test | 66,164 | 248,149,734 bytes | 86,706,292 | 4.71% |
| 合計 | 1,405,297 | 5,324,931,609 bytes | 1,858,455,821 | 100.00% |

全行についてUTF-8 JSON、`id`、`source_page_id`、`title`、`url`、`text`、本文100文字
以上、全splitを通したID一意性、pageのsplit非重複、split規則を検証済みである。
metadataのSHA-256も元dumpから再計算した値と一致した。学習側本文ローダーで3 splitの
全件を読み、各splitから32 valid tokenの少量batchを生成できた。train本文は約16.86億
文字あり、15,000,000 tokenの事前学習予算に対して十分な規模である。

再生成時の確定コマンドは次のとおり。本環境では`/dev/shm`が64MiBに固定されているため、
内容を変えず共有blobのbacking storeだけを通常ファイルへ移す互換層とそのソースも
`data/tools/`に保持している。

~~~sh
sha256sum data/raw/jawiki-20260901/jawiki-20260901-pages-articles-multistream.xml.bz2

(
  cd data/tools/WikiExtractor
  LD_PRELOAD=../shm_file_backend.so python3 -m wikiextractor.WikiExtractor \
    --json --processes 8 --namespaces 0 \
    --output ../../../data/intermediate/jawiki-20260901 \
    ../../../data/raw/jawiki-20260901/jawiki-20260901-pages-articles-multistream.xml.bz2
)

scripts/prepare_jawiki.py data/intermediate/jawiki-20260901 data/jawiki-20260901 \
  --dump-file data/raw/jawiki-20260901/jawiki-20260901-pages-articles-multistream.xml.bz2 \
  --dump-date 20260901 \
  --dump-url https://dumps.wikimedia.org/jawiki/20260901/jawiki-20260901-pages-articles-multistream.xml.bz2 \
  --dump-sha256 888a2a81e7e2888f364953a69d5fc55b88449df8977eb2be01316a4c53da7252 \
  --extractor-revision e96bee708b6a0f3ee66a4b677294faa80e3557a6 \
  --min-chars 100 --max-documents 0
~~~

準備スクリプトは元dumpのSHA-256を検証し、抽出器revision、段落条件、split規則、件数、
dump URL、[Wikimedia利用規約](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en)
をmetadata.jsonへ保存する。page IDのhashで90/5/5へ分けるため、段落がsplitをまたがない。
再配布時は元記事の帰属情報を保持し、学習済み重み公開前にも適用ライセンスを確認する。

## 共通tokenizerと学習経路

共通BPEは会話trainとjawiki trainだけから一度作る。validation/testは渡さない。

~~~sh
build/quality/main_train tokenizer \
  data/conversation/train.jsonl data/jawiki-20260901/train.jsonl \
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
TEXT_DATA=data/jawiki-20260901 \
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
