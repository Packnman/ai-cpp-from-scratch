# `include/ai/agent/memory.h` 設計書

## 目的

`SqliteMemory`はSemantic/Episodic/Project記憶をSQLiteへ永続化し、日本語検索を提供する。
`MemoryRetrieveTool`は同機能をAgentのtool契約へ適合させる。

## 保存契約

- contentは非空、最大16 KiB。importance/confidenceは0以上1以下。
- `(type, content)`を一意キーとしてupsertし、importance/confidenceは最大値を保持する。
- `store_batch`は`BEGIN IMMEDIATE`のtransactionで全件成功またはrollback。
- timestampはSQLiteの`unixepoch()`を使用する。

## 検索契約

- limitは最大8。空queryまたはlimit 0は空結果。
- 3 code point以上はFTS5 trigramとBM25、短いqueryはparameter bindingした完全・部分一致。
- type filterは任意。取得行の`last_accessed_at`をtransactionで更新する。
- FTS queryは引用符をescapeしてphrase検索する。

## DBと互換性

WAL、foreign key、5秒busy timeout、`SQLITE_OPEN_FULLMUTEX`を使用する。schema versionは1。
version 0からmigrateし、実装より新しいversionは拒否する。copyは禁止し、destructorで
connectionを閉じる。

## Tool契約

入力は`query`文字列、任意の`type`と`limit`。入力不正はPermanentError、DB例外は
RetryableError、成功時は公開可能な記憶fieldのJSON配列を返す。

## 実装・検証

- 実装: `src/agent/memory.cpp`
- テスト: `tests/agent/agent_check.cpp`, `tests/agent/memory_migration_check.cpp`
