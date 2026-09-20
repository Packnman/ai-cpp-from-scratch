# Motor Variant / Selection 詳細設計
# 1. Selected Windings
| Class | Selected Article | Reason |
| :- | :- | :- |
| S | 22ECT35 10B 80 01 | 24 V系で高Kt 27.31 mNm/A、Continuous Current 0.7 A |
| M | 22ECT48 10B 35 01 | 24 V系で高Kt 28.08 mNm/A、Continuous Current 1.5 A |
| L | 22ECT60 10B 21 01 | 24 V系で高Kt 25.97 mNm/A、Continuous Current 2.6 A |
# 2. Alternative Windings
高速側のActuatorが必要になった場合は同一Familyの低Kt / 高Speed巻線へ変更可能とする。
Prototype 1では部品共通化と低～中速Torqueを優先しBaselineを固定する。
# 3. Application Mapping
```text
22ECT35-80
├── Sternocleidomastoid
├── Splenius capitis
├── Biceps brachii
├── Deltoid
├── Serratus anterior
└── Trapezius
22ECT48-35
├── Pectoralis major
├── Latissimus dorsi
├── Rectus femoris
├── Biceps femoris
└── Adductor magnus
22ECT60-21
├── Gluteus maximus
└── Waist Linear Cylinder
```
# 4. Change Rule
巻線変更は以下のいずれかが成立した場合のみ行う。
- Required actuator speed不足
- Thermal margin不足
- Driver current / voltage constraint
- Transmission ratioが非現実的
- Mass / package constraint
変更時はActuator Power Budget、Driver Parameter、Transmission Ratioを再評価する。
