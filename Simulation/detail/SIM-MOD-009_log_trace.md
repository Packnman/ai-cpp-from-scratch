# SIM-MOD-009 Logging

`SimulationLog` は snapshot を CSV に追記する。列は simulation time、joint position/velocity/acceleration/effort、contact body pair と force で、ヘッダを一度だけ出力する。再現性を損なう壁時計 timestamp は使用しない。

ログ先の所有とローテーションは呼出側の責務である。I/O failure は stream state として検出できる。
