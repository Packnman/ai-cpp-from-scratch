# Context builder 設計仕様

対象ヘッダ: `model/include/context_builder.h`。実装: `model/src/context_builder.cpp`。

## 責務と公開API

`context_builder` は、長い会話や文書から原文の完全な単位を選び、token予算内のpromptを構築する。生成型要約や外部検索は行わず、学習時と推論時で同じ選択規則を共有する。

- `ConversationContextTurn`: 質問と回答からなる完全な1往復。
- `BuiltContext`: 構築済みtoken列と、採用した履歴ターン・文書文の元index。
- `g_buildConversationContext`: 完全な過去QAと現在質問から会話promptを作る。
- `g_splitDocumentSentences`: UTF-8文書を原文を保ったまま文単位へ分ける。
- `g_buildDocumentContext`: 関連文書文、直近QA、現在質問から文書QA promptを作る。

## token予算と必須領域

会話builderは、次の必須領域を履歴より先に確保する。

```text
BEGIN
SPEAKER_A current_question UTTERANCE_END
SPEAKER_B
```

必須領域だけで上限を超える場合は `std::invalid_argument` を返す。token列、UTF-8文字、文、発話、ターンを途中で切断せず、現在質問は常に完全な形で残す。

生成CLIがbuilderへ渡す入力予算は `input_context - max_tokens` である。文脈長1024、既定出力256の場合は768 tokenとなる。生成・学習側の予算予約は [会話runtime](conversation_runtime.md) と [会話dataset](dataset_conversation.md) を参照する。

## 会話履歴の選択

履歴容量は直近履歴60%と質問関連履歴40%を基準に選択する。

1. 質問と回答が完全一致する重複ターンは最新だけを残す。
2. 最新の完全ターンが総容量に収まる場合、60%枠を超えても先に確保する。
3. 新しいターンから順に60%枠を埋める。
4. 未選択ターンをBM25スコア順に残り容量へ追加する。
5. 関連候補で余った容量を未選択の直近履歴へ移譲する。
6. 選択後は元index順へ戻し、会話の時系列を保つ。

各ターンは次の不可分なtoken単位であり、質問または回答だけを採用しない。

```text
SPEAKER_A question UTTERANCE_END
SPEAKER_B response UTTERANCE_END
```

## BM25と決定性

質問と候補を同じ `TokenConversation` でencodeし、話者・区切りなどの特殊IDは語頻度から除外する。定数は `k1 = 1.2`、`b = 0.75` とする。候補集合内の文書頻度、候補token長、平均候補長からスコアを求める。

同点時は会話ターンでは新しい元index、文書文では小さい元indexを優先する。乱数を使わないため、tokenizer、入力、予算が同じなら選択結果も同じになる。

## 文書分割と文書context

空文書は拒否し、`TokenCharacter` でUTF-8妥当性を検証する。境界は `。`、`！`、`？`、`.`、`!`、`?`、改行であり、境界文字自体は直前の文へ残す。完全一致する文は最初の出現だけを候補にする。

文書モードでは必須領域を除く容量の60%へ新しい完全QAを入れ、残容量へBM25上位の文書文を入れる。文書文で余った容量は未選択の直近QAへ移譲する。選択後は文書文と履歴をそれぞれ元の順序へ戻す。prompt形式は次のとおり。

```text
BEGIN
selected_document_sentences UTTERANCE_END
selected_complete_QA_turns
SPEAKER_A current_question UTTERANCE_END
SPEAKER_B
```

## エラー条件と検証

空文書、不正UTF-8、現在質問だけでの予算超過を入力エラーとする。長すぎるターンや文は単位全体を不採用とし、部分切断はしない。

`tests/context_builder_check.cpp` は、token上限、最新・関連ターン選択、最新重複保持、決定性、予算超過拒否、日本語・ASCII・改行境界、数値needleを含む文のBM25選択を検証する。
