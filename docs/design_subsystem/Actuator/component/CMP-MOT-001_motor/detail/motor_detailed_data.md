# Detailed Motor Data
# 1. Baseline Article
| Parameter | S: 22ECT35 10B 80 01 | M: 22ECT48 10B 35 01 | L: 22ECT60 10B 21 01 |
| :- | --: | --: | --: |
| Nominal Voltage | 24 V | 24 V | 24 V |
| Motor Diameter | 22 mm | 22 mm | 22 mm |
| Motor Length | 35 mm | 48 mm | 60 mm |
| Number of Poles | 4 | 4 | 4 |
| Feedback | Hall Sensors | Hall Sensors | Hall Sensors |
| Hall Electrical Phasing | 120° | 120° | 120° |
| No-Load Speed | 8,100 rpm | 7,950 rpm | 9,180 rpm |
| Typical No-Load Current | 40 mA | 70 mA | 115 mA |
| Max Continuous Mechanical Power @25°C | 34 W | 54 W | 86 W |
| Max Continuous Current | 0.7 A | 1.5 A | 2.6 A |
| Max Continuous Torque | 19.5 mNm | 40.8 mNm | 64.3 mNm |
| Torque Constant Kt | 27.31 mNm/A | 28.08 mNm/A | 25.97 mNm/A |
| Back EMF Constant | 2.86 V/1000 rpm | 2.94 V/1000 rpm | 2.72 V/1000 rpm |
| Phase-Phase Internal Resistance | 9.2 Ω | 2.4 Ω | 1.08 Ω |
| Line-Line Resistance | 9.23 Ω | 2.43 Ω | 1.11 Ω |
| Phase-Phase Inductance | 0.75 mH | 0.24 mH | 0.123 mH |
| Electrical Time Constant | 0.08 ms | 0.10 ms | 0.11 ms |
| Mechanical Time Constant | 4.4 ms | 1.9 ms | 1.4 ms |
| Rotor Inertia | 3.6 gcm² | 6.3 gcm² | 8.71 gcm² |
| Motor Constant | 9.0 mNm/W^0.5 | 18.1 mNm/W^0.5 | 25.0 mNm/W^0.5 |
| Maximum Motor Speed | 20,000 rpm | 20,000 rpm | 20,000 rpm |
| Maximum Continuous Speed | 20,000 rpm | 20,000 rpm | 20,000 rpm |
| Thermal Resistance | 2.3 / 13 °C/W | 2.1 / 12 °C/W | 2.0 / 8.8 °C/W |
| Thermal Time Constant | 829 s | 962 s | 980 s |
| Maximum Winding Temperature | 125°C | 125°C | 125°C |
| Ambient Operating Range | -30 to +100°C | -30 to +100°C | -30 to +100°C |
| Ambient Storage Range | -40 to +100°C | -40 to +100°C | -40 to +100°C |
| Max Axial Static Force without Shaft Support | 45 N | 45 N | 45 N |
| Ball Bearing Preload | 6.8 N | 6.8 N | 6.8 N |
| Housing | Stainless Steel / Aluminium Flange | Stainless Steel / Aluminium Flange | Stainless Steel / Aluminium Flange |
| Mass | 67 g | 98 g | 123 g |
# 2. Torque Conversion
Motor torque estimate:
```text
tau_motor ≈ Kt * I
```
Example at 1.0 A:
```text
22ECT35-80 : ≈ 27.31 mNm
22ECT48-35 : ≈ 28.08 mNm
22ECT60-21 : ≈ 25.97 mNm
```
Actual usable current must respect each winding's continuous current, driver limit, temperature and duty cycle.
# 3. Design Implication
Torque Constantが近いため、Current-to-Torque conversionを共通化しやすい。
一方、許容Continuous CurrentとThermal CapacityはMotor長により異なるため、同一Current Limitを全Classへ適用してはならない。
# 4. Data Usage Restriction
- No-load speedを負荷時Speedとして扱わない。
- Max continuous torqueとMax continuous powerを同時成立する点として扱わない。
- 25°C Continuous RatingをRobot内部高温環境へそのまま適用しない。
- Axial static force 45 NはScrew thrust capacityとして利用せず、Screw axial loadはTransmission Bearingで支持する。
# 5. Source
Portescap公式Product Page / Specification Page。
詳細リンクは `manufacturer_links.md` を参照する。
