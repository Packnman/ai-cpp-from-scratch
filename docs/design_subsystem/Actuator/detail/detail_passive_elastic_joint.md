# Passive Elastic Joint 詳細設計

# 1. 目的

Motorを使用せずSpring / Damperにより着地衝撃、高周波振動を吸収し、地面追従性を得る。初期対象はAnkle。

# 2. モデル

```text
tau = -k * theta - c * theta_dot
```

必要に応じ非線形Spring、Mechanical Stopを追加する。

# 3. 注意

Spring単体はエネルギーを蓄積・返却するが、振動エネルギーを散逸しない。振動吸収を目的とするためDamperを組み合わせる。

# 4. 状態監視

- ankle angle
- angular velocity
- spring deflection
- estimated spring torque
- mechanical stop contact

# 5. Safety

- over deflection
- mechanical stop collision
- damper degradation
- spring break suspicion

# 6. 未確定

- k
- c
- neutral angle
- travel
- nonlinear profile
