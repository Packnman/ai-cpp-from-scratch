# BRN-MOD-007 Policy Manager 詳細設計書

> 親文書: `Brain System モジュール詳細設計 README`
>
> 本書は、親文書で定義した共通型、所有関係、Event / Queueベース実行モデル、初期実装方針、ディレクトリ構成、実装フェーズおよびCodex実装制約に従う。


# 1. 目的

Policy Manager はPlannerが利用するAction Templateや行動方策を管理する。

# 2. Policy

```cpp
struct Policy
{
    PolicyId id;
    std::string name;

    ConditionExpression applicableCondition;
    ActionTemplate actionTemplate;

    int priority;
    float confidence;
    float successRate;

    PolicySource source;
    std::uint32_t version;
};
```

# 3. Policy Type

- Predefined
- Learned
- ExperienceDerived
- DomainSpecific

# 4. Applicable判定

```text
Goal
+
WorldState
+
Condition
↓
ApplicableCondition.evaluate()
```

# 5. Selection

候補スコア:

```text
score =
    priority
  + confidence
  + context_match
  + success_rate
```

# 6. Interface

```cpp
class IPolicyManager
{
public:
    virtual ~IPolicyManager() = default;

    virtual void add(const Policy&) = 0;
    virtual void update(const Policy&) = 0;

    virtual std::vector<Policy> findApplicable(
        const PlanningContext&
    ) const = 0;
};
```

# 7. 初期実装

初期Policyは設定ファイルまたはコード定義のRule-based Policyとする。

例:

```text
Goal = AcquireObject
→ Locate
→ Approach
→ Reach
→ Grasp
→ Verify
```

# 8. Post-process連携

Action Resultから:

- successRate
- confidence
- usageCount

を更新可能とする。

# 9. Safety

PolicyがCritical Constraintに反する場合は採用しない。

# 10. Version

Policy変更ごとにVersionを増加させる。

# 11. Unit Test

- Applicable
- priority
- confidence
- success rate update
- Constraint violation
- version

# 12. 完了条件

- Rule-based Policy登録・検索
- Plannerへ候補供給
- Resultによる統計更新
