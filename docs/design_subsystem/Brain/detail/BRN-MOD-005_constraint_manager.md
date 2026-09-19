# BRN-MOD-005 Constraint Manager 詳細設計書

> 親文書: `Brain System モジュール詳細設計 README`
>
> 本書は、親文書で定義した共通型、所有関係、Event / Queueベース実行モデル、初期実装方針、ディレクトリ構成、実装フェーズおよびCodex実装制約に従う。


# 1. 目的

Constraint Manager はBrainの行動に適用されるHard / Soft Constraintを一元管理する。

# 2. Constraint

```cpp
struct Constraint
{
    ConstraintId id;
    ConstraintType type;

    bool critical;
    bool active;

    ConstraintScope scope;
    ConstraintExpression expression;

    ConstraintSource source;

    std::optional<TimePoint> expiresAt;
    TimePoint timestamp;
};
```

# 3. Source Priority

```text
Safety
>
System Internal Rule
>
Human Explicit
>
Environment
>
Policy
```

# 4. Scope

```cpp
enum class ConstraintScopeType
{
    Global,
    Goal,
    Plan,
    Action,
    Resource,
    Entity
};
```

# 5. Hard Constraint

`critical == true`

違反時:

- Candidate Plan Reject
- Running Plan Re-evaluation
- 必要ならExecution Cancel

# 6. Soft Constraint

`critical == false`

Plan ScoreへPenaltyとして反映する。

# 7. Interface

```cpp
class IConstraintManager
{
public:
    virtual ~IConstraintManager() = default;

    virtual void add(const Constraint&) = 0;
    virtual void update(const Constraint&) = 0;
    virtual void remove(ConstraintId) = 0;

    virtual std::vector<Constraint> active() const = 0;

    virtual ConstraintEvaluation evaluate(
        const ActionPlan& plan
    ) const = 0;
};
```

# 8. Expiration

Event Loop周期で期限切れConstraintをinactiveへ遷移する。

# 9. Conflict

相互矛盾するConstraintが存在する場合:

1. Source Priority
2. critical
3. timestamp
4. 解決不能ならPlanning Failure

# 10. Safety Change

Safety Constraint追加時はPlannerへ `ReplanRequired` Eventを発行する。

# 11. Thread Safety

Active Constraint Storeはread-heavyのため `shared_mutex` を使用可能。

# 12. Unit Test

- Add/update/remove
- Hard
- Soft
- Scope
- Expiration
- Conflict
- Safety override
- Re-plan trigger

# 13. 完了条件

- Critical Constraint違反Planを通さない
- Expiration可能
- Safety Constraint変更を即時反映
