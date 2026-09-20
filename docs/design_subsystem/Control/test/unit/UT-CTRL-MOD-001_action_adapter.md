# CTRL-MOD-001 Action Adapter 単体テスト仕様
# 1. 目的
CTRL-MOD-001 Action Adapterを他Moduleから分離し、Mock / Fake依存で責務を検証する。
# 2. 基本Test
- initialize成功
- valid input正常処理
- invalid input拒否
- boundary condition
- fault input
- timeout / missing input
- reset / stop
# 3. Module固有確認
対象Moduleの入力・出力・State Transition・Error Handlingが詳細設計どおりであることを確認する。
# 4. Mock
必要に応じBrain / Sensor / Actuator / Safety / PowerをMock化する。
# 5. Acceptance
- unexpected exceptionなし
- invalid outputなし
- state transition整合
- fault時に安全側へ遷移
