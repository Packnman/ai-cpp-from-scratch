# Protection 詳細設計
Hardware over-currentをSoftwareより優先する。
Critical faultはGate Disable。
Command timeout時はOutputを安全側へ遷移する。
