# Thermal Design 詳細設計
# 1. Manufacturer Limits
| Parameter | 22ECT35 | 22ECT48 | 22ECT60 |
| :- | --: | --: | --: |
| Max Winding Temperature | 125°C | 125°C | 125°C |
| Ambient Operating | -30 to +100°C | -30 to +100°C | -30 to +100°C |
| Thermal Time Constant | 829 s | 962 s | 980 s |
| Thermal Resistance | 2.3 / 13 °C/W | 2.1 / 12 °C/W | 2.0 / 8.8 °C/W |
# 2. Robot Design Rule
Prototypeでは125°CをControl Targetにしない。
Warning / Derating / Stop thresholdはMarginを持って別途設定する。
# 3. Thermal Estimation
```text
P_cu ≈ I_rms^2 * R
```
Motor Current履歴からCopper Lossを推定し、First-order thermal modelでWinding Temperatureを推定可能とする。
# 4. Bundle Effect
筋肉Actuator内で複数Motorを近接配置するため、単体Datasheet Thermal ResistanceだけでなくBundle実測Temperatureを用いて補正する。
# 5. Validation
- continuous low-load
- nominal gait duty
- intense motion duty
- stalled / near-stalled protection
- adjacent motor simultaneous drive
をThermal Test対象とする。
