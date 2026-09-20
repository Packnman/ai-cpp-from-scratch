# Control Test README
# 1. Unit Test
`unit/` 配下にCTRL-MOD-001～010の単体テスト仕様を配置する。
# 2. Integration Test
`integration/IT-CONTROL_control_integration_test.md` にControl System結合テストを配置する。
# 3. 方針
単体TestではModule責務を分離して検証し、結合TestではBrain/Sensor/Actuator/Safety/PowerをMock/Fake化してPipeline全体を検証する。
