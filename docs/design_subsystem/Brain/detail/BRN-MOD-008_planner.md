# BRN-MOD-008 Planner 詳細設計書

> 親文書: `Brain System モジュール詳細設計 README`
>
> 本書は、親文書で定義した共通型、所有関係、Event / Queueベース実行モデル、初期実装方針、ディレクトリ構成、実装フェーズおよびCodex実装制約に従う。


# 1. 目的

Planner はActive GoalとWorld Stateをもとに実行可能なAction Planを生成し、必要時に再計画する。

# 2. Planner構成

```text
IPlanner
├── RuleBasedPlanner        [初期必須]
├── TransformerPlanner      [Stub可]
└── ExternalAIPlanner       [Stub可]
```

# 3. PlanningContext

```cpp
struct PlanningContext
{
    Goal goal;
    WorldState worldState;

    std::vector<Constraint> constraints;
    std::vector<MemoryItem> memories;
    std::vector<Policy> policies;

    std::optional<ActionResult> previousResult;
};
```

# 4. IPlanner

```cpp
class IPlanner
{
public:
    virtual ~IPlanner() = default;

    virtual PlanningResult plan(
        const PlanningContext&
    ) = 0;
};
```

# 5. RuleBasedPlanner

処理:

1. Goal Type判定
2. Applicable Policy取得
3. Action Template展開
4. Preconditions生成
5. Resource割当
6. Critical Constraint評価
7. Action Plan生成
8. Validate
9. Return

# 6. Action

```cpp
struct Action
{
    ActionId id;
    ActionType type;

    std::optional<SemanticId> target;
    AttributeMap parameters;

    std::vector<ConditionExpression> preconditions;
    std::vector<ConditionExpression> completionConditions;
    std::vector<ConditionExpression> failureConditions;

    Duration timeout;
    int priority;

    std::vector<ResourceRequest> resources;
};
```

# 7. ActionPlan

```cpp
struct ActionPlan
{
    PlanId id;
    GoalId goalId;

    std::vector<ActionNode> actions;
    std::vector<ConstraintId> constraints;

    PlanStatus status;
    TimePoint createdAt;

    std::uint64_t worldStateVersion;
    std::uint32_t version;
};
```

# 8. Validation Pipeline

```text
Generated Plan
↓
Schema Validation
↓
Dependency Validation
↓
Precondition Validation
↓
Resource Validation
↓
Critical Constraint Check
↓
Accept
```

# 9. Preconditions

例:

```text
Grasp(ball)
requires:
    visible(ball)
    reachable(ball)
    hand_free(right)
    robot_stable
```

不成立時:

- prerequisite Action追加
- Candidate reject
- Planning failure

# 10. Plan Score

RuleBasedPlannerでは複雑な連続最適化を必須としない。

将来の候補評価:

```text
score =
    + goalAchievement
    - safetyRisk
    - executionTime
    - energy
    - uncertainty
```

# 11. Re-planning

Trigger:

- Goal Changed
- Target Moved
- Condition Changed
- Constraint Changed
- Action Failed
- Timeout
- Safety State Changed
- Communication State Changed

# 12. TransformerPlanner

初期はStubでよい。

将来入力:

```text
Goal Token
World Tokens
Condition Tokens
Constraint Tokens
Memory Tokens
Policy Tokens
Action History Tokens
```

出力は必ずValidation Pipelineを通す。

# 13. ExternalAIPlanner

External AI Adapter越しに呼び出す。

Timeout時はRuleBasedPlannerへFallbackする。

# 14. Planning Timeout

Planner全体にもTimeoutを持つ。

初期Fallback:

```text
Transformer / External
→ RuleBasedPlanner
→ Failure
```

# 15. File構成

```text
include/brain/planning/
├── Planner.hpp
├── PlanningTypes.hpp
├── RuleBasedPlanner.hpp
├── TransformerPlanner.hpp
├── ExternalAIPlanner.hpp
└── PlanValidator.hpp
```

# 16. Unit Test

- simple goal
- multi action
- unmet precondition
- hard constraint
- resource conflict
- invalid dependency
- replan
- external timeout fallback
- deterministic plan

# 17. 完了条件

- RuleBasedPlannerが実動作
- Plan Validationが機能
- Constraint違反Planを出さない
- 外部Planner失敗時Fallback可能
