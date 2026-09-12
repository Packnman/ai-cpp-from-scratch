# 会話学習・保存・生成 API 設計仕様

対象ヘッダ: `model/include/conversation_runtime.h`（保存）、`train/include/conversation_training.h`（学習）、`validation/include/conversation_validation.h`（評価・生成）。C++20／CUDA 実装。CLI は `main_train` と `main_validation chat`。

## 公開 API と設定

`g_trainConversation(dataDir, modelDir, modelConfig, trainingConfig, log)` は3分割を読み、モデルを構築し、学習・最良保存・test 評価まで実行する。

| ConversationTrainingConfig | 既定値 |
| --- | --- |
| nEpochs | 10 |
| nBatchSize | 64 |
| fLearningRate | 3e-4 |
| fClipNorm | 1.0 |
| nSeed | 42 |
| nMaxBatches | 0（全件、正数なら各 split/epoch をその batch 数で制限） |

Optimizer は既存 Adam（beta1=0.9、beta2=0.999、epsilon=1e-8）。FP32。train seed をモデル初期化・dropout・window shuffle に使用する。CLI の `--seed` は非負 int の範囲。分割の seed は常に42。

`g_finetuneConversation(dataDirectory, sourceModelDirectory, outputDirectory, trainingConfig, log)` は保存済み bundle を読み、同じ内部学習ループで指定回数の追加 epoch を実行する。重み・モデル構造・文脈長・dropout・モデル seed・語彙・特殊 ID は保存値を維持する。語彙は再構築せず、全 split の未知文字を UNK にする。Adam の moment と step は新規初期化する。追加学習の training seed は shuffle のみに使用し、dropout は保存モデル seed から新しい乱数列を開始する。以前の乱数状態は復元しない。

CLI は `main_train DATA OUTPUT --from-model SOURCE --batch 32 --epochs 10 --max-batches 0`。`--from-model` なしの新規学習は従来どおり。追加学習と構造オプション（blocks、embedding、heads、hidden、context、dropout）の併用は、保存値と同じ値でも拒否する。batch／lr／clip／epochs／max-batches は変更可能。

出力先は元モデルとは異なる新規または空のディレクトリに限る。正規化したパスで同一性を確認し、元モデルへの symlink、非空ディレクトリ、通常ファイルを拒否する。無効な bundle は出力作成前に拒否する。保存先を他プロセスが同時変更する運用は契約外。

追加学習は更新前の validation を epoch 0 として評価し、読み込んだ重みを最初の最良候補として保存する。その後の各 epoch で厳密な改善がなければ元の重みが残る。最後に最良重みを読み直して test を評価する。`max-batches` の制限は更新前の validation にも適用する。

標準出力と `metrics.jsonl` の `training_start` イベントに読み込み元、Adam 新規初期化、追加 epoch 数、batch、lr、clip、shuffle/dropout seed、構造と dropout、batch 制限を記録する。指標行は従来の形式で、追加 epoch は1から数える。保存は bundle v1 のままで、再読込・再追加学習・対話生成に使用できる。これは重みからの追加学習であり、Optimizer・乱数状態を含む完全な中断再開ではない。

`g_generateConversation(model, tokenizer, history, rng, generationConfig = {})` は生成 ID 列（停止 ID を含む）を返す。

| ConversationGenerationConfig | 既定値 |
| --- | --- |
| fTemperature | 0.8 |
| nTopK | 40 |
| nMaxTokens | 128（1～128） |

`g_clipGradients(model, maximum)` は clip 前の全体 L2 norm を返す。`g_saveConversation` は bundle を書き、`g_loadConversation` は所有モデル unique_ptr と tokenizer を持つ `ConversationBundle` を返す。

## 学習順序・形状・数式

対話 ID の3分割間非重複確認→ train 本文のみから語彙構築→各 split を window 化→各 epoch の train shuffle→zero grads→ `[T,B]` IDs の forward→有効 token 平均 loss→backward→全 Parameter の `norm=sqrt(Σθ ||dθ||²)` を cuBLAS で計算→ `dθ*=min(1,limit/norm)`→Adam。

validation は dropout 無効。epoch loss は batch loss を有効 token 数で重み付けし、`perplexity=exp(loss)`。validation loss が厳密に改善したときに保存する。終了時は保存した最良重みを再ロードし test を評価する。

`metrics.jsonl` と指定 log に epoch、split、loss、perplexity、有効 token、batch 数、秒、tokens/s、CUDA 使用メモリの各 batch 終了時サンプル最大、baseline、batch 制限を記録する。メモリ値には他プロセスの使用量が含まれ、CUDA 一時メモリの厳密な peak ではない。nMaxBatches 正数での指標は部分データの動作確認であり全件評価ではない。

## 生成順序

利用者 A の本文を encode し、`A,本文,UTTERANCE_END,B` を履歴へ追加。直近 `min(128,config.nContext)` token を入力し、最後の位置の logits を使う。END／UTTERANCE_END／通常文字だけを候補とし、PAD、UNK、BEGIN、A、B を除外する。top-k に絞り、`exp((logit-max)/temperature)` を重みとして sample。停止記号で終了する。KV cache は使用せず毎 token 再計算。API は一時的に評価モードに切り替え、例外時を含め元に戻す。

CLI は END で履歴を BEGIN に戻す。最大長で停止した場合は次の利用者入力前に発話終了を補う。履歴は128へ切り詰める。標準入力 EOF で終了する。

## 所有権・保存形式・互換性

学習 API の内部モデル・Optimizer は関数が所有する。Dataset は CPU window、batch 転送はローカル GPU Tensor。loss/graph は各 batch の末尾に解放する。生成の model/tokenizer/rng/history は借用し、生成結果は値で返す。返却 bundle のモデルは bundle 所有で元データ不要。

ディレクトリ内 `weights.bin` は既存 Model v2 の名前・shape 付き FP32 Parameter、`manifest.json` は format=`ai_cpp_conversation`、version=1、dtype=float32、モデル全設定、昇順通常語彙、全特殊 ID。load は format/version/dtype/特殊 ID/語彙とモデルサイズを検証する。weights の名前・shape 検証は Model に委譲する。Optimizer、dropout RNG、shuffle 状態は保存しない。重みからの推論は再現できるが完全な学習再開ではない。

weights を先に、manifest を最後に書く。複数ファイルをまとめた atomic commit ではないため、保存中の同時読込・中断後の bundle 利用は契約外。別設定で同じ出力ディレクトリを同時に使用しない。

## 例外・完了条件

空 split／ID 重複、不正設定、I/O、manifest 不整合、CUDA 失敗、非有限 loss／勾配／生成 logits を拒否する。temperature、lr、clip は有限正数、batch/epoch/top-k は正数。異常は CLI が stderr に出し終了コード1。

`conversation_check` で短系列過学習、save/load の logits 完全一致、禁止 ID を高 logit にした生成の除外と停止、train→validation→best test を確認する。全既存テストと、指定構成の実データ実測は [実測記録](conversation_validation.md) を参照。

## 実行ファイルと単独評価

入口は `train/main_train.cpp` と `validation/main_validation.cpp`。引数処理はそれぞれの `src/*_cli.cpp`、学習・評価 API は各 `include/` と `src/` に配置する。Transformer／tokenizer／dataset／bundle 保存は `model/include/`・`model/src/` で共有する。

`main_train DATA MODEL [OPTIONS]` で新規／追加学習、`main_train prepare SOURCE OUTPUT [REVISION]` で前処理を実行する。`main_validation DATA MODEL [--split validation|test --batch N --max-batches N]` は保存重みを単独評価し、`main_validation chat MODEL [OPTIONS]` は対話生成する。

`g_validateConversation(dataDirectory, modelDirectory, split, batchSize, maxBatches, log)` は保存モデルを評価モードで読み、指定 split の有効 token 数で重み付けした平均 loss を返す。JSON ログに loss／perplexity／有効 token 数／batch 数／batch size／batch 制限／読み込み元を出力する。UNK と PAD の扱いは学習時と同じ。split は validation または test、batch は正数、maxBatches は非負。空 split・不正設定・非有限 loss を拒否する。Optimizer 初期化・更新・モデル書き込みは行わない。
