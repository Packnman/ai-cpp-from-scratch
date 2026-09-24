# IT-BRAIN Brain System 単体結合テスト仕様書

# 1. 目的

BRN-MOD-001〜012をBrain System内部で結合し、外部Robot Hardwareを使用せず、Brain単体として入力からGoal生成、Planning、Execution、Post-process、Memory、Traceまでの一連の処理が成立することを確認する。

# 2. Test Double

- FakeSensorSource
- FakeHMI
- FakeSafetySystem
- MockControlSystem
- FakeExternalAI
- FakeClock
- Temporary SQLite DB
- MemoryLogBackend

# 3. 対象フロー

```text
Input
→ Pre-process
→ World State / Goal / Constraint
→ Planner
→ Execution Manager
→ Mock Control
→ Action Result
→ Post-process
→ Memory / Policy / Goal Update
→ Log / Trace
```

# 4. 基本Action Set

```text
Locate(target)
MoveTo(target)
Reach(target)
Grasp(target)
Release(target)
Put(target, destination)
Open(target)
Close(target)
Wait()
Stop()
```

# 5. 基本Goal Set

```text
FindObject
AcquireObject
MoveObject
PutObject
AnswerQuestion
StopCurrentAction
```

# 6. 必須結合テスト

| ID | Scenario | 期待結果 |
| :- | :- | :- |
| IT-001 | 「青いボールを取って」+ visible/reachable | Goal→Reach→Grasp→Achieved |
| IT-002 | target invisible | Locateを含むPlan |
| IT-003 | target unreachable | MoveToを含むPlan |
| IT-010 | Critical Constraint違反 | Plan採用なし / Dispatchなし |
| IT-011 | Soft Constraint違反 | Plan可 / penalty |
| IT-012 | 実行中にSafety Constraint追加 | Re-plan / 必要ならCancel |
| IT-020 | 高Priority Goal追加 | Current Suspend / Preemption |
| IT-030 | Sequential actions | 順序通りDispatch |
| IT-031 | Parallel non-conflict | 並列実行 |
| IT-032 | Exclusive Resource conflict | 同時実行なし |
| IT-033 | Action Timeout | Cancel→Timeout→Re-plan |
| IT-040 | External AI success | Validate後反映 |
| IT-041 | External AI timeout | Local RuleBased fallback |
| IT-042 | Late External response | 現Planへ反映しない |
| IT-050 | Action success memory | STM/LTM更新 |
| IT-051 | Previous failure retrieval | Planner Contextへ取得 |
| IT-052 | SQLite failure | STM-onlyでBrain継続 |
| IT-060 | Stale perception | 古いvisibleを確定利用しない |
| IT-061 | Target move | WorldState version更新→Re-plan |
| IT-070 | Recognition failure | BrainError / fallback |
| IT-071 | Planner failure | Dispatchなし / Goal failure等 |
| IT-072 | Control failure | Post-process→Memory→Re-plan |
| IT-080 | EStop before plan | 新規Dispatchなし |
| IT-081 | EStop during execution | Running停止要求 / 自動Resumeなし |
| IT-090 | Sensor burst + Safety event | Safety優先 |
| IT-091 | Queue overflow | Policy通り / Safety飢餓なし |
| IT-100 | End-to-end Trace | Goal→Plan→Action→Result追跡可能 |
| IT-110 | Determinism | 同一入力/状態/Seed→同一結果 |

# 7. 代表Scenario詳細

## IT-001 AcquireObject

入力:

```text
Text:
    青いボールを取って

World State:
    blue_ball.visible = true
    blue_ball.reachable = true
    right_hand.free = true
    robot.stable = true
```

期待:

```text
Goal = AcquireObject(blue_ball)

Plan:
    Reach(blue_ball)
    Grasp(blue_ball)

Result:
    Achieved
```

確認項目:

- Input AdapterがTextを受付
- Context RecognitionがGoal生成
- Goal ManagerがActive化
- Plannerが有効Plan生成
- Executionが順番通りDispatch
- Mock Control Resultを受信
- Post-processがGoal達成判定
- Memoryへ成功経験保存
- TraceがGoal/Plan/Action/Resultを関連付ける

## IT-010 Critical Constraint

```text
Goal:
    MoveTo(kitchen)

Constraint:
    do_not_enter(kitchen)
    critical = true
```

期待:

- Plan invalid
- Execution Dispatch = 0
- Constraint violation記録

## IT-033 Action Timeout

Mock Controlを無応答にする。

期待:

```text
Running
→ Timeout
→ Cancel
→ Resource Release
→ Re-plan Trigger
```

## IT-041 External AI Timeout

期待:

```text
External AI
→ Timeout
→ Local RuleBasedPlanner
→ Brain継続
```

## IT-081 EmergencyStop

実行中にEStop投入。

期待:

- Safety Eventを優先処理
- 新規Dispatch停止
- Running ActionへStop
- 自動Resumeなし
- Trace記録

# 8. Regression Test Data

```text
tests/data/brain/
├── context.jsonl
├── planning.jsonl
├── constraint.jsonl
├── execution.jsonl
└── integration.jsonl
```

例:

```json
{
  "name": "acquire_visible_reachable_ball",
  "input": {
    "text": "青いボールを取って"
  },
  "world_state": {
    "blue_ball.visible": true,
    "blue_ball.reachable": true,
    "right_hand.free": true,
    "robot.stable": true
  },
  "expected": {
    "goal": "AcquireObject",
    "plan": [
      "Reach(blue_ball)",
      "Grasp(blue_ball)"
    ],
    "result": "Achieved"
  }
}
```

# 9. OSS Datasetの位置付け

Unit / Brain Integrationの合否は自作の決定論的Datasetを基本とする。

OSS Datasetは別Benchmark層として利用する。

- JGLUE: 日本語理解
- CLINC150: Intent
- ALFRED: Goal / Action Plan
- AI2-THOR: World State / Execution
- MultiWOZ: Conversation / Memory
- BEHAVIOR: 長期統合Benchmark

# 10. 評価指標

```text
Goal Success Rate
Plan Valid Rate
Constraint Violation Count
Action Success Rate
Re-plan Success Rate
Timeout Recovery Rate
External AI Fallback Success Rate
Memory Retrieval Accuracy
Trace Completeness
```

# 11. 合格基準

初期Brain Core:

- Critical Constraint violation = 0
- EmergencyStop中の新規Dispatch = 0
- Dependency violation = 0
- Exclusive Resource collision = 0
- Late External AI Response誤適用 = 0
- Trace欠落 = 0
- Unit Test全件PASS
- 必須Integration Scenario全件PASS
- DB / External AI / Recognition単一障害でBrain Coreが異常終了しない

# 12. 実行順序

```text
1. 全Unit Test
2. Basic Integration
3. Constraint / Safety
4. Failure / Timeout
5. Memory
6. External AI
7. Queue Stress
8. Regression
9. OSS Benchmark
```

# 13. 完了条件

Mock環境でBrain単体の

```text
Input
→ Recognition
→ State / Goal / Constraint
→ Planning
→ Execution
→ Result
→ Post-process
→ Memory
→ Trace
```

を再現可能であり、正常系・異常系・Safety・Timeout・Fallbackの必須Scenarioが全件成功すること。
