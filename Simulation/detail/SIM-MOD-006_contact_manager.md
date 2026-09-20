# SIM-MOD-006 Contact Manager

`MuJoCoContactManager` は `mjData::contact` と `mj_contactForce` を logical body pair、world contact point、normal/tangent force に変換する。contact geom から body を引き、同一 body 内 contact も ID を保持する。

- Output units: position m, force N
- Empty case: contact vector is empty
- Test: `UT-SIM-MOD-006_contact_manager.cpp`
