# `include/ai/agent/tools.h` 設計書

## `CalculatorTool`

入力`{"expression": string}`を、副作用のない再帰下降parserで評価する。対応構文は有限の
10進数、空白、単項`+/-`、括弧、`+ - * /`。関数呼出し、識別子、非有限値、ゼロ除算は
拒否する。成功値は`{"value": number}`、入力・計算不正はPermanentError。

## `FileReadTool`

constructorで許可rootをcanonical化し、directoryでなければ拒否する。実行時は相対path
だけを受け、canonical化後もroot配下であることをcomponent単位で検査する。

読み取り対象はregular file、最大1 MiB、有効なUTF-8に限定する。path traversal、絶対
path、binary/不正UTF-8はPermanentError、読み取りI/O失敗はRetryableError。成功時は
要求された相対pathとcontentを返す。

## 非責務

shell実行、書込み、network access、任意精度計算は提供しない。tool登録とoperation名の
解決は`ToolRegistry`の責務。

## 実装・検証

- 実装: `src/agent/tools.cpp`
- テスト: `tests/agent/agent_check.cpp`
