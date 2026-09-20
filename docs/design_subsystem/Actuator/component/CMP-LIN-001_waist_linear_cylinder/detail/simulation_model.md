# CMP-LIN-001 Simulation Model

`SimWaistLinearCylinder` は22ECT60 presetとScrew Transmissionの1本分のCompositionである。
length、velocity、estimated force、current、temperature、stroke limit、faultを返す。
左右2本の協調制御は上位Control/Actuator層の責務であり、本Componentには含めない。

Screw lead / ratio / efficiency、stroke、最大推力は腰機構解析後に確定するためConfiguration値とする。

