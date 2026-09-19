# BRN-MOD-012 Log / Trace Manager 詳細設計書

> 親文書: `Brain System モジュール詳細設計 README`
>
> 本書は、親文書で定義した共通型、所有関係、Event / Queueベース実行モデル、初期実装方針、ディレクトリ構成、実装フェーズおよびCodex実装制約に従う。


# 1. 目的

Log / Trace Manager はBrain Systemの入力、判断、状態遷移、Error、Version、外部AI通信等を記録し、障害解析・再現試験・学習データ生成を支援する。

# 2. TraceEvent

```cpp
struct TraceEvent
{
    TraceId id;
    TraceType type;

    TimePoint timestamp;

    std::optional<GoalId> goalId;
    std::optional<PlanId> planId;
    std::optional<ActionId> actionId;

    ModuleId source;

    AttributeMap data;
};
```

# 3. BrainError

```cpp
struct BrainError
{
    ErrorId id;
    ErrorLevel level;
    ModuleId module;

    TimePoint timestamp;

    std::string description;
    std::string cause;

    RecoveryAction recovery;
};
```

# 4. Correlation

以下を一連のTraceとして紐付ける。

```text
Goal
└── Plan
    ├── Action
    │   └── Result
    └── Action
        └── Result
```

# 5. Log Level

```cpp
enum class LogLevel
{
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical
};
```

# 6. Backend

```cpp
class ILogBackend
{
public:
    virtual ~ILogBackend() = default;

    virtual void write(const TraceEvent&) = 0;
    virtual void flush() = 0;
};
```

初期候補:

- JSONL
- SQLite

# 7. LogManager

```cpp
class ILogManager
{
public:
    virtual ~ILogManager() = default;

    virtual void log(const TraceEvent&) = 0;
    virtual void report(const BrainError&) = 0;
};
```

# 8. 非同期書込

Runtime PathをBlockしないため、Log Queueを用意可能とする。

```text
Producer
↓
Log Queue
↓
Log Worker
↓
Backend
```

初期実装は同期でもよいが、Interfaceは非同期化可能にする。

# 9. 記録対象

- Input
- Semantic output
- WorldState version
- Goal
- Constraint
- Memory query/result
- Policy selection
- Planning context
- Candidate plan
- Selected plan
- Action command/result
- External AI
- Safety state
- Error
- Version

# 10. Version記録

- Software
- Model
- Configuration
- Representation
- DB Schema
- Policy

# 11. Raw Sensor

高頻度Raw Sensor DataはBrain Logへ常時保存しない。

Sensor Systemの専用Logへ委譲する。

# 12. Backend Failure

- Warning
- Memory buffer
- Drop low priority logs
- Critical log loss通知

Brain Coreを停止しない。

# 13. Replay

将来的にTraceから以下を再現可能にする。

- Goal
- PlanningContext
- selected Plan
- Action Result

# 14. Unit Test

- correlation
- log levels
- backend write
- backend failure
- queue overflow
- critical preservation
- version output
- replay metadata

# 15. 完了条件

- Goal/Plan/Action追跡可能
- Backend交換可能
- Log障害でBrain停止なし
- Version情報記録可能
