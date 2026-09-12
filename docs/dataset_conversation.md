# 会話データセット設計仕様

対象ヘッダ: `model/include/dataset_conversation.h`。

## 責務・API

C++20 と vendored nlohmann/json 3.12.0 を用い、公式 RealPersonaChat JSON の本文を対話単位で分割・読み込み・window 化する。Python は不要。

`g_prepareConversations(sourceRepo, outputDirectory, revision = "unspecified")`、`g_readConversations(file)`、`g_trainingText(conversations)`、`ConversationDataset(conversations, tokenizer, nContext = 128)`、`size()`、`batch(order, start, batchSize)` を公開する。

`Conversation` は int64 ID と発話列、`ConversationUtterance` は対話内話者番号0/1と UTF-8 本文を持つ。ペルソナ・属性・評価・timestamp はモデル入力へ渡さない。

## 前処理と形式

公式取得元: [nu-dialogue/real-persona-chat](https://github.com/nu-dialogue/real-persona-chat)。`SOURCE/real_persona_chat/dialogues/*.json` を読む。ID で整列後 seed 42 の mt19937 と rejection sampling の Fisher-Yates でシャッフルする。train=`floor(N*0.90)`、validation=`floor(N*0.05)`、test=残り。分割後にのみ window を作る。

`train.jsonl`、`validation.jsonl`、`test.jsonl` は UTF-8 で1行1対話、schema は `{"id":整数,"utterances":[{"speaker":0または1,"text":"本文"}]}`。`metadata.json` は format=`ai_cpp_conversations_v1`、取得元、VERSION、指定 revision、seed、分割アルゴリズム、各 split の全 ID を含む。話者は interlocutors の順に A/B へ対応し、発話の最初の出現順には依存しない。

## 系列・Tensor 形状・所有権

対話ごとに BEGIN → (話者 → 本文 → UTTERANCE_END)* → END を作る。最大 C+1 個を C ずつ進めて切り出す。`X[t,b]=window[t]`、`Y[t,b]=window[t+1]`、残りは双方 PAD。最後の1 token だけで新 window を作らない。対話をまたぐ教師は作らない。

Dataset は int32 window を値で所有し、構築後は元対話と tokenizer を破棄できる。batch は CPU の row-major `[C,actualBatch]` 入力・教師と有効教師数を返し、呼出し側が GPU へコピーする。端数バッチの actualBatch は残件数。末尾の短系列も C まで右 PAD を行う。GPU 上の PAD は損失平均の分母に含めない。本文・両話者の区切り・対話終端を教師にする。

## モード・勾配・例外

非微分の整数処理。学習時のみ呼出し側が window 順をシャッフルし、評価順は固定。語彙構築は train の本文のみ。

I/O／JSON／UTF-8 の不正、重複 ID、未知話者、空対話、不正な設定・batch 範囲・window index は例外とする。分割には20件以上を要求する。学習 API は3 split 間の全 ID 重複も再確認する。schema 変更には format version の更新が必要。

## 完了条件

`conversation_check` の40対話で36/2/2の分割、再実行一致、分割間非重複、話者 ID 対応、教師の1位置ずれ、window 間 stride、対話境界、右 PAD、端数バッチを確認する。公式実データへの結果は [実測記録](conversation_validation.md) に記載する。

### 公式データ中の重複

同一 dialogue_id が複数ファイルにある場合、話者対応と本文が完全に一致するものは分割前に1対話へ統合し、metadata の `deduplicated_files` に ID とファイル名を記録する。同じ ID で本文が異なる場合は例外とする。前処理後 JSONL の重複 ID は引き続き拒否する。`conversation_check` は同一内容の重複ファイルを与えても40対話の分割数が変わらないことを確認する。

短い最終 window では入力に END 自体も含め、その位置の教師は PAD とする。たとえば C=8、`[BEGIN,A,UTTERANCE_END,END]` の入力は4 token＋4 PAD、教師は `[A,UTTERANCE_END,END]`＋5 PAD、有効教師数は3。
