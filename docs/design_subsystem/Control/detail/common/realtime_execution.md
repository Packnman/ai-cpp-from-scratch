# Realtime Execution
# 1. Cycle
```text
Feedback Read
→ State Update
→ Safety/Power Constraint
→ Controller Update
→ Joint Allocation
→ Command Output
→ Monitor
```
# 2. Timing
各ModuleはCycle Timeを計測しDeadline Missを検出する。
制御周期はTBD。
# 3. Overrun
```text
Warning
→ lower-priority computation skip
→ controller degradation
→ SafeStop request
```
