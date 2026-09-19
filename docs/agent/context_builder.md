# `include/ai/agent/context_builder.h` 設計書

## 目的

`ContextInput`の構造化情報から、固定予算内のモデルpromptを決定的に構築する。
token数の測り方は`Counter`注入によりcode point概算と実tokenizerを切り替える。

## 必須section

`Current Input`、`Goal`、`Current Task`、criticalな`Critical Constraints`は必ず入る。
`require_summary=true`では`Conversation Summary`も必須である。必須部分だけで予算を
超えた場合は`length_error`とし、黙って情報を削除しない。

## 任意sectionの優先順

1. `Grounded Entity Candidates`
2. 指定範囲の`Recent Conversation`
3. `Retrieved Memory`
4. 必須でない`Conversation Summary`
5. `Previous Result`

各sectionは全体が収まる場合だけ追加する。NER候補には「確定事実ではない」と明記し、
種類、UTF-8 byte span、表記、任意の正規化値、抽出元を渡す。

## 補助API

- `structured_prompt_input`: mode用payloadとschema、1回限りのrepair指示を組み立てる。
- `build_model_prompt`: `ContextBuilder`を簡便に呼ぶ自由関数。
- `utf8_codepoints`: fallback計数。token数と同一とは限らない。

## 実装・検証

- 実装: `src/agent/context_builder.cpp`
- テスト: `tests/agent/agent_check.cpp`, `tests/model/agent_model_check.cpp`
