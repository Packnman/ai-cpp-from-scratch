# BRN-MOD-003 World State Manager 詳細設計書

> 親文書: `Brain System モジュール詳細設計 README`
>
> 本書は、親文書で定義した共通型、所有関係、Event / Queueベース実行モデル、初期実装方針、ディレクトリ構成、実装フェーズおよびCodex実装制約に従う。


# 1. 目的

World State Manager は、現在の外界状態・Robot State・Condition・Communication State・Safety Stateを統合し、Plannerへ一貫したSnapshotを提供する。

# 2. WorldState

```cpp
struct WorldState
{
    std::uint64_t version;
    TimePoint timestamp;

    std::unordered_map<SemanticId, Perception> perceptions;
    RobotState robotState;
    std::unordered_map<ConditionId, Condition> conditions;

    CommunicationState communicationState;
    SafetyState safetyState;
};
```

# 3. 責務

- SemanticItemの反映
- Entity統合
- Stale管理
- Conflict管理
- Snapshot生成
- WorldState Version更新

# 4. Interface

```cpp
class IWorldStateManager
{
public:
    virtual ~IWorldStateManager() = default;

    virtual void update(const SemanticItem& item) = 0;
    virtual WorldState snapshot() const = 0;

    virtual std::uint64_t version() const = 0;
};
```

# 5. Entity Resolution

同一Entity判定優先順位:

1. 明示SemanticId
2. TrackingId
3. Relation
4. Spatial proximity
5. Class / feature similarity

自信がない場合は無理に統合しない。

# 6. Update Rule

```text
New observation
    ↓
Validate timestamp
    ↓
Find entity
    ↓
Compare old/new confidence
    ↓
Update / Keep / Conflict
    ↓
Increment WorldState version
```

# 7. Stale管理

各TypeにTTLを持つ。

Staleになった情報は即削除せず `valid=false` 相当で残せる設計とする。

# 8. Snapshot

Snapshot生成中にUpdateされてもPlanner側のSnapshot内容は変化しないよう、copyまたはimmutable viewを使用する。

初期実装はcopyでよい。

# 9. Thread Safety

UpdateとSnapshotは排他制御する。

初期実装候補:

```cpp
mutable std::shared_mutex _mutex;
```

- update: unique lock
- snapshot: shared lock

# 10. Conflict

例:

```text
object.position = A
object.position = B
```

同Timestamp近傍で矛盾する場合:

- Confidence比較
- Source優先度比較
- Conflict Flag設定
- 必要なら再認識Trigger

# 11. Safety State

Safety Stateは一般Perceptionより高優先で更新する。

Stale Safety Stateは安全側へ扱う。

# 12. Log

- Version
- Entity add/update/remove
- Conflict
- Stale transition
- Safety State change

# 13. ファイル構成

```text
include/brain/world/
├── WorldState.hpp
└── WorldStateManager.hpp

src/brain/world/
└── WorldStateManager.cpp
```

# 14. Unit Test

- Add entity
- Update entity
- Tracking統合
- Conflict
- Stale
- Snapshot consistency
- Concurrent update/read
- Version increment

# 15. 完了条件

- 一貫したWorldState Snapshotが取得可能
- Entity重複を抑制
- Stale情報を識別可能
- Concurrent accessで破損しない
