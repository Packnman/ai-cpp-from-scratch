# CMP-SPR-001 Ankle Passive Spring-Damper Component 詳細設計 README
# 1. 目的
本書はComponent設計仕様書を実装可能な粒度へ展開する詳細設計の親文書である。
# 2. 詳細文書
- [Spring / Damper 詳細設計](./spring_damper.md)
- [Mechanical Interface 詳細設計](./mechanical_interface.md)
- [Simulation Model](./simulation_model.md)
- [Parameter Identification 詳細設計](./parameter_identification.md)
# 3. 実装原則
- 上位要求仕様書を満たすこと。
- Component境界外のSystem責務を持ち込まないこと。
- TBDは測定・解析・選定根拠とともに確定すること。
- Safety Limitは上位Commandより優先すること。
