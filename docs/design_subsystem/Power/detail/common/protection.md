# PWR-DTL-006 Protection
# 1. Protection階層
```text
Level 0  Motor Driver local protection
Level 1  Branch protection
Level 2  Actuator Bus protection
Level 3  Main Battery / BMS protection
```
# 2. Over Current
```text
Warning
→ Software derating
→ Driver current limit
→ Branch cut
→ Main cut
```
# 3. Under Voltage
Logic Rail維持を優先し、Actuator Branchを段階的に停止する。
```text
Neck / non-critical upper
→ Upper task
→ Waist non-essential
→ Lower non-stance
→ SafeStop
```
# 4. Thermal
Temperature WarningでPower Limit、CriticalでBranch Disableとする。
# 5. Short Circuit
Hardware protectionを必須とし、Softwareのみへ依存しない。
# 6. Recovery
Fault種類ごとにAuto Recovery可否を設定する。
Short Circuit、Battery Fault、Main Overcurrentは原則Manual Resetとする。
