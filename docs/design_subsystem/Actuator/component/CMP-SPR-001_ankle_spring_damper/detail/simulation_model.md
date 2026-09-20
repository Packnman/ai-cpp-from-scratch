# CMP-SPR-001 Simulation Model

`SimSpringDamper` は能動Drive Commandを持たないPassive Adapterである。

```text
tau = -k * (theta - theta_neutral) - c * theta_dot
```

角度がmin/maxを越えた場合は設定された有限mechanical stop stiffnessによる反力を追加する。
k、c、neutral、travel、stop stiffnessは30 kg Robotの試験・同定前のためConfiguration値とする。

