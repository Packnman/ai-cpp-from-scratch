# `include/ai/agent` 設計書一覧

| ヘッダー | 設計書 | 主責務 |
|---|---|---|
| `agent.h` | [agent.md](agent.md) | 1ターンのオーケストレーション |
| `components.h` | [components.md](components.md) | component interfaceと既定実装 |
| `context_builder.h` | [context_builder.md](context_builder.md) | prompt構築と予算制御 |
| `discussion.h` | [discussion.md](discussion.md) | 根拠付き限定比較と訂正状態 |
| `memory.h` | [memory.md](memory.md) | SQLite長期記憶 |
| `reasoner.h` | [reasoner.md](reasoner.md) | rule/model/hybrid backend |
| `tools.h` | [tools.md](tools.md) | 計算・制限付きファイル読取 |
| `types.h` | [types.md](types.md) | Agent層の共有データ型 |
