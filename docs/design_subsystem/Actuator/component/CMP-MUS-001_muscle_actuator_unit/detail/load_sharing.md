# Multi-Motor Load Sharing 詳細設計
# 1. Goal
Motor間のTorque / Current偏りを抑える。
# 2. Method
Common shaftならTorque sharing、independent screwならposition synchronization + current balancingを使用する。
# 3. Fault
1 Motor異常時はGroup outputをDeratingし、必要に応じGroup停止する。
