# BRN-MOD-011 External AI Adapter 詳細設計書

> 親文書: `Brain System モジュール詳細設計 README`
>
> 本書は、親文書で定義した共通型、所有関係、Event / Queueベース実行モデル、初期実装方針、ディレクトリ構成、実装フェーズおよびCodex実装制約に従う。


# 1. 目的

External AI Adapter は外部計算資源へAI処理を委譲し、Request / Response / Timeout / Retry / Fallback / Context整合を管理する。

# 2. Request Type

```cpp
enum class ExternalTaskType
{
    HighLevelRecognition,
    LanguageInference,
    KnowledgeQuery,
    HeavyPlanning,
    Embedding,
    ModelSpecificTask
};
```

# 3. Request

```cpp
struct ExternalAIRequest
{
    RequestId id;
    ExternalTaskType type;

    TimePoint createdAt;
    Duration timeout;

    Payload payload;

    GoalId goalId;
    std::optional<PlanId> planId;
    std::uint64_t worldStateVersion;

    std::uint32_t schemaVersion;
};
```

# 4. Response

```cpp
struct ExternalAIResponse
{
    RequestId requestId;
    ResponseStatus status;

    TimePoint receivedAt;

    Payload payload;

    std::optional<float> confidence;

    std::string modelId;
    std::string modelVersion;
};
```

# 5. Request State

```cpp
enum class ExternalRequestState
{
    Created,
    Sending,
    Waiting,
    Succeeded,
    Timeout,
    Failed,
    Cancelled
};
```

# 6. Interface

```cpp
class IExternalAI
{
public:
    virtual ~IExternalAI() = default;

    virtual RequestId submit(
        const ExternalAIRequest&
    ) = 0;

    virtual std::optional<ExternalAIResponse> poll(
        RequestId
    ) = 0;

    virtual void cancel(RequestId) = 0;
};
```

# 7. Context Validation

Response受信時:

- RequestId一致
- GoalIdがまだ有効
- WorldStateVersion差が許容範囲
- Timeoutしていない
- Schema互換
- Payload valid
- Safety Constraint違反なし

# 8. Late Response

Timeout済みRequestのResponseは自動採用しない。

必要ならCacheまたはLogへ保存する。

# 9. Retry

初期方針:

- Idempotent requestのみ自動Retry
- 最大Retry回数はConfig
- exponential backoffは将来拡張可

# 10. Fallback

```text
External AI
↓ timeout/failure
Local Small Model
↓ unavailable
Rule Based
↓ impossible
Failure / Clarification / SafeStop
```

# 11. FakeExternalAI

初期実装必須。

同一Requestに同一Responseを返す決定論的Fakeを用意する。

# 12. Connection State

- Offline
- Connecting
- Online
- Degraded

通信状態はCommunication Systemから受け取る。

# 13. Log

- Request ID
- Type
- Goal
- WorldStateVersion
- Send time
- Response time
- Model version
- Timeout
- Retry
- Fallback

# 14. Unit Test

- normal response
- timeout
- retry
- late response
- wrong ID
- stale WorldState
- invalid schema
- fake determinism
- fallback

# 15. 完了条件

- FakeExternalAIで全経路試験可能
- TimeoutでBlockしない
- Late Response誤適用なし
- Local Fallback可能
