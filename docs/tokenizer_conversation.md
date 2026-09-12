# TokenConversation 設計仕様

対象ヘッダ: `model/include/tokenizer_conversation.h`。

## 責務・API・設定

train 本文だけを受け取る `TokenConversation( std::string_view )` は `TokenCharacter` を再利用する。`encode`、`decode`、`vocabulary()`、`vocabSize()` を公開する。vocabulary は通常文字だけを昇順に1回ずつ含む UTF-8 文字列。

| 特殊定数 | ID | 用途 |
| --- | --- | --- |
| PAD | 0 | 右 padding、損失除外 |
| UNK | 1 | train にないコードポイント |
| BEGIN | 2 | 対話開始 |
| END | 3 | 対話終了 |
| SPEAKER_A | 4 | 話者 A |
| SPEAKER_B | 5 | 話者 B |
| UTTERANCE_END | 6 | 発話終了 |
| SPECIAL_COUNT | 7 | 通常 ID のオフセット |

通常 ID は `TokenCharacter ID + 7`。本文の文字列に特殊記号の表記があっても専用 ID として解釈しない。`decode` は区切りを表示せず、UNK だけ U+FFFD を表示する。生成候補は END、UTTERANCE_END、通常文字。

## データ・所有権・処理・例外

TokenCharacter を値として所有し、返却文字列／int32 ID ベクトルは呼出し側が所有する。GPU Tensor はなく、勾配なし。encode は UTF-8 検証と lookup を行い、未知コードポイント1個につき UNK 1個を出す。不正 UTF-8 は `invalid_argument`、範囲外 ID は `out_of_range`。学習／評価で同じ固定語彙を使う。validation/test で語彙を追加しない。

## 保存形式・互換性・完了条件

会話 bundle v1 の manifest に通常語彙文字列と特殊 ID の全対応を保存する。読込時は特殊 ID の完全一致と語彙の昇順・一意性、モデル語彙数を検証する。元対話データは不要。

`conversation_check` で日本語、改行、4 byte Unicode、未知文字、特殊 ID と通常文字の分離、保存再読込の logits 一致を確認する。
