# CTRL-MOD-001 Action Adapter
# 責務
Brain SystemからActionCommandを受信し、Control内部形式へ正規化する。
# 入力
- ActionCommand
# 出力
- NormalizedAction
# 処理
- schema validation
- action type validation
- target validation
- timeout / priority normalization
- unsupported action rejection
# Error
Invalid ActionはExecution MonitorへFailedとして通知する。
# Dependency
Brain Interface、Motion Manager。
