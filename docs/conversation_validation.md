# 会話 Transformer 検証記録

検証日: 2026-09-12。実装・前処理・学習・生成は C++20／CUDA。Python は使用しない。

## 環境

- NVIDIA GeForce RTX 3060 Ti、8 GiB、CUDA architecture 86
- CUDA Toolkit 12.8.93、ホスト C++ compiler GCC 13.3
- CMake Release、Ninja、ビルド先 `/tmp/ai-cpp-rpc/build`
- JSON: 同梱 nlohmann/json 3.12.0（MIT）

## 自動テスト

`ctest --test-dir /tmp/ai-cpp-rpc/build --output-on-failure` で、既存14件と新規 `conversation_check` を確認する。新規テストの内容:

- 40対話の36/2/2分割、同 seed の再現性、分割間 ID 非重複、同一対話の重複統合
- 対話内 ID による話者対応、日本語・改行・4 byte Unicode、未知文字と strict tokenizer 互換性
- 1 token 先の教師、window の stride、対話境界、短系列の右 PAD、端数バッチ
- 整数教師の log-sum-exp 参照と勾配、PAD 除外、全 PAD、教師 snapshot、不正 ID
- 2ヘッド／2バッチの Attention 入力および Q/K/V/O 全 Parameter の中心差分（epsilon=0.001、絶対誤差0.002以内）
- 未来入力を変更しても過去出力が不変、別 shape の forward を保持してからの backward
- 別 shape の forward を挟んでも過去の dropout の backward が同 seed の独立実行と一致
- 1ブロック、E16、H2、F32、4 token の固定系列へ80回 Adam 更新し、loss が `2.94191 → 0.000163794`
- 保存前後の FP32 logits が完全一致、禁止 ID の高 logit を除外して発話終了で停止
- train → validation → 最良保存 → 最良重みで test の API 一貫動作

## 公式データの前処理

取得元: [RealPersonaChat 公式リポジトリ](https://github.com/nu-dialogue/real-persona-chat)。VERSION=`1.0.0`、revision=`28d0b6b3865b29cabc26c230a2db37cdf315e937`。

```sh
/tmp/ai-cpp-rpc/build/conversation_prepare \
    /tmp/ai-cpp-rpc/source /tmp/ai-cpp-rpc/data \
    28d0b6b3865b29cabc26c230a2db37cdf315e937
```

元の13,583 JSON 中、同一 ID・同一入力本文の重複が2件あった（ID 10577、11838）。分割前に統合して13,581対話とし、train 12,222、validation 679、test 680。統合ファイル名と全 split ID を `metadata.json` に記録した。train 語彙は特殊 ID を含め3,197、train window は77,742。

## 指定構成での GPU 実測

4ブロック、E256、4ヘッド、head size64、FFN1024、T128、B8、FP32。Adam lr=0.0003、dropout=0.1、全体 norm 上限1.0、seed42。

```sh
/tmp/ai-cpp-rpc/build/conversation train \
    /tmp/ai-cpp-rpc/data /tmp/ai-cpp-rpc/model \
    --epochs 1 --max-batches 3
```

各 split を3バッチに制限した動作確認の結果。train は forward・backward・clip・Adam update を含む。速度には batch 転送と logging 用集計を含め、モデル構築・JSON 読込・語彙構築は含めない。warmup の除外は行っていない。

| split | 有効 token | loss | perplexity | 秒 | 有効 token/秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,782 | 8.041837 | 3,108.32 | 0.499730 | 5,567.01 |
| validation | 2,806 | 7.040585 | 1,142.06 | 0.271760 | 10,325.28 |
| test（最良保存後） | 2,940 | 7.076728 | 1,184.09 | 0.268707 | 10,941.27 |

train の baseline CUDA 使用量は1,203,240,960 bytes、各バッチの更新後に計算グラフを保持した状態で測った最大使用量は1,591,214,080 bytes（約1.48 GiB）、baseline との差は370 MiB。validation/test のサンプル最大は1,551,368,192 bytes。`cudaMemGetInfo` によるデバイス全体の測定であり、他プロセス分も含む。演算途中の一時確保を含めた厳密な peak ではない。

## 保存・再読込・返答生成

`/tmp/ai-cpp-rpc/model` の `weights.bin` と `manifest.json` だけを渡す `conversation chat` を別プロセスで起動し、元データを引数に与えず返答を生成した。

```sh
/tmp/ai-cpp-rpc/build/conversation chat /tmp/ai-cpp-rpc/model --max-tokens 16
```

入力「こんにちは」に対し16 token を生成し、次の利用者入力待ちへ戻った。生成はランダムな文字列であり、3更新時点で意味のある返答品質には達していない。停止記号と禁止 ID の動作は上記の決定的な自動テストで別途確認した。

全10 epoch の全件学習、全件 test perplexity、対話品質評価は未実施。ここでの実データ指標は動作と資源使用の検証であり、モデル品質の評価値としては使用しない。

## 保存モデルからの追加学習（2026-09-12）

`cmake --build build/release -j2` と `ctest --test-dir build/release --output-on-failure` を実行し、追加学習の検証を拡張した `conversation_check` を含む全15件が成功（6.67秒）。小規模 fixture で次を確認した。

- 読み込み元と更新前 logits が完全一致し、epoch 0 validation が独立計算の loss と一致
- 保存語彙にない追加データの文字が UNK になり、manifest（構造・文脈長・dropout・語彙・特殊 ID・モデル seed）が変わらない
- shuffle seed=99 と保存モデル seed=17 の分離、B32・max-batches=0 で全3バッチを2追加 epoch 学習
- validation loss が `2.76409 → 0.679768` に減少し、重みが更新され、再読込した重みの loss が最良値と一致
- 空ディレクトリへの再追加学習と再読込、更新が丸め落ちる微小 lr で改善がない場合の初期候補保持
- 元 bundle のバイト単位の不変性、同一出力・symlink 別名・非空出力・通常ファイル・不正モデルの拒否

CLI の blocks／embedding／heads／hidden／context／dropout の6オプションを `--from-model` と併用して、それぞれ終了コード1、構造変更を拒否するエラー、出力が作成されないことを確認した。

実データの追加学習は既存 `models/conversation`（4ブロック、E256、4ヘッド、FFN1024、T128、dropout0.1、seed42、語彙3197）を読み込み、RTX 3060 Ti 8 GiB で実施した。

```sh
build/release/conversation train data/conversation /tmp/ai-cpp-finetune-zz6m85f3/model \
    --from-model models/conversation --batch 32 --epochs 1 --max-batches 3
build/release/conversation chat /tmp/ai-cpp-finetune-zz6m85f3/model --max-tokens 16
```

Adam は新規初期化、lr=0.0003、clip=1。各 split を3バッチに制限した動作確認であり、全件追加学習や品質評価ではない。

| split | loss | 有効 token | 秒 | 有効 token/秒 |
| --- | ---: | ---: | ---: | ---: |
| 更新前 validation（epoch 0） | 1.887755 | 11,358 | 0.701289 | 16,195.90 |
| 追加 train（epoch 1） | 1.751753 | 11,264 | 0.813921 | 13,839.18 |
| validation（epoch 1） | 1.889600 | 11,358 | 0.532729 | 21,320.40 |
| test_best | 1.863838 | 11,680 | 0.551353 | 21,184.25 |

train のサンプル最大 GPU 使用量は2,698,510,336 bytes（約2.51 GiB）、baseline は1,213,726,720 bytes。デバイス全体の測定で他プロセス分を含み、演算途中の厳密な peak ではない。validation が改善しなかったため、出力の最良重みは読み込み元とバイト単位で一致した。元モデル内ファイルの SHA-256 が実行前後で変わらないことも確認した。別プロセスの chat で「こんにちは」に対し「こんにちは」を生成して次の入力待ちに戻った。実行ログは上記出力の親ディレクトリの `train.log` と `chat.log` に保存した。

## model／train／validation への分割後の検証

共通コードを `model/include/`・`model/src/`、学習コードを `train/include/`・`train/src/`、評価・生成を `validation/include/`・`validation/src/` へ配置した。入口は `train/main_train.cpp` と `validation/main_validation.cpp` の2つ。上の実測にある `conversation train` は現在の `main_train`、`conversation chat` は `main_validation chat`、`conversation_prepare` は `main_train prepare` に対応する。

Release ビルドと CTest 全15件が成功（5.76秒）。単独評価の loss が学習中 validation の最良値と一致し、バッチサイズを32から1に変えても有効 token で重み付けした loss が一致することを追加検証した。不正 split、batch、max-batches も拒否する。

小規模40対話 fixture で新 CLI の prepare → 新規学習 → 追加学習 → validation／test 評価 → chat を別プロセスで実行して成功した。評価・生成の前後で保存先全ファイルの SHA-256 は不変。構造変更オプション、不正 split、batch0、引数不足は終了コード1になった。画像モデルを使用する既存の基盤テストは `tests/fixtures/` を参照して引き続き成功している。
