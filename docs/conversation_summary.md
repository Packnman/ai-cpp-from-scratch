# 内部会話要約

runtime と SFT は次の6セクションをこの順序で使います。空欄は `- なし` です。

```text
[FACTS]
[DECISIONS]
[CONSTRAINTS]
[OPEN_QUESTIONS]
[ACTIVE_TASKS]
[CORRECTIONS]
```

CHAT の入力予算は 766 token（1024 - BOS/mode 2 - 出力予約256）です。
EVALUATE と MEMORY_QUERY は出力128、入力894 token、それ以外は出力256、入力766 tokenです。
CHAT は固定入力、critical constraints、構造化要約を必須とし、残りへ直近2ターン、検索memory、先行結果の順に追加します。
全履歴を含む次回CHATが入力予算を超えると、直近2ターンを残す最大prefixを要約します。
要約は192 tokenを目標、256 tokenを上限とし、タグ順、UTF-8、長さを検証します。
不正時の修復は1回だけで、再失敗時は旧要約も会話履歴も削除しません。

## データ生成

元コーパスや変換済み本文はコミットしません。利用者がライセンスと版を確認してローカル配置します。
JMultiWOZ は CC BY-SA 4.0 の revision を必ず指定します。

```sh
python3 scripts/convert_summary_sft.py jmultiwoz \
  --input /local/JMultiWOZ/train.json --output /tmp/jmultiwoz-train.jsonl \
  --revision COMMIT_OR_DATASET_REVISION --split train

python3 scripts/convert_summary_sft.py mix \
  --base data/agent-v3-base-sft/train.base.jsonl \
  --jmultiwoz /tmp/jmultiwoz-train.jsonl \
  --output data/agent-v3-base-sft/train.jsonl --seed 42
```

`mix` は既存 SUMMARIZE 件数を保ち、JMultiWOZ 70%、決定的合成データ30%を選びます。
各JSONL行は従来の `id/mode/prompt/output/sft_profile/split/generation_seed` に加えて、
`source_name/source_id/source_revision/converter_version` を持ちます。変換manifestは採用数と除外理由を記録します。
SFT loader もcontext超過例を切り詰めず除外し、model manifestの `sft_exclusions` に件数を残します。

Hugging Face の `llm-book/livedoor-news-corpus` ローダー（revisionを固定）から取得でき、`url/date/title/content/category` のJSON/JSONL exportと元アーカイブ展開ディレクトリの両方を入力にできます。
ライブドアニュースはタイトルを FACTS のみの弱教師にする比較用です。本文要約データとして既定学習へは入れません。

```sh
python3 scripts/convert_summary_sft.py livedoor \
  --input /local/livedoor-train.jsonl --output /tmp/livedoor-5.jsonl \
  --revision ARCHIVE_CHECKSUM --split train --mix-percent 5

python3 scripts/convert_summary_sft.py mix \
  --base data/agent-v3-base-sft/train.base.jsonl \
  --jmultiwoz /tmp/jmultiwoz-train.jsonl --livedoor /tmp/livedoor-5.jsonl \
  --livedoor-percent 5 --output /tmp/agent-livedoor-5.jsonl
```

商用モデルへ採用する前に BY-ND 条件の権利確認が必要です。変換例には
`rights_review_required: true` が記録されます。`dialogsum-ja` と PoliInfo2 は既定入力に含めません。

## 評価

学習テンプレートと固有名詞を共有しない100会話を人手で用意し、数値、否定、訂正、制約、
未解決事項、反復要約を各20件以上含めます。各variantを同じtokenizer、初期checkpoint、更新数で3 seed実行します。
結果JSONLは回答正解率、カテゴリ別/macro F1、1/3/5回保持率、prompt token数、圧縮率、
構造失敗率、要約欠落率、overflow率、validation loss、bits-per-byteを含めます。

```sh
python3 scripts/evaluate_summary_ablation.py \
  --input results/summary-ablation.jsonl --output results/summary-report.json
```

採用判定は、品質低下を全体1ポイント・カテゴリ2ポイント以内に抑えたうえで、回答正解率かmacro F1を
1ポイント以上改善する、または両品質を維持してprompt tokenを5%以上削減することです。
全量SFT前に1 shard相当を実行し、checkpoint再開、非有限lossなし、8 GiB環境での実行を確認します。
