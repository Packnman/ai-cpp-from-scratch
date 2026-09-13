# 回答中心損失の実装・検証

実装日: 2026-09-13

`main_train --loss-target all|response` を追加した。既定の `all` は従来互換で、
`response` は各 `SPEAKER_B` 回答の本文と末尾 `UTTERANCE_END` だけを損失対象にする。
直前のA質問とB回答は同じサンプルに固定し、余裕があれば古い完全なA/Bターンを
新しい順に収まるだけ前置する。context超過時は古いターン単位で除き、直前Q/Aだけでも
収まらない回答は分断せず除外する。

開始イベントには `loss_target` と train の `excluded_responses`、epoch checkpointには
`training.loss_target` を保存する。再開時は保存方式を復元し、明示された方式が異なれば
重み更新前に停止する。validation/testも学習と同じマスクを使うため、response方式の
loss/perplexityは応答対象tokenだけの値である。

## 決定的fixture

`response_training_check` は8件の短い日本語Q/Aを使用し、次を検証する。

- QとAが常に同一サンプルにあり、古い履歴は完全ターンとしてのみ追加される。
- 非PAD targetが回答本文と末尾の発話終端だけである。
- 単一Q/Aがcontextを超える例は除外され、件数が増える。
- 2層・embedding 32・4 heads・FFN 64の小型モデルを同じ8件へ500 step適合する。
- `top-k 1` で8問すべてから対応回答と発話終端を再現する。

RTX 3060 Tiでの実測は loss `3.50887 -> 3.03984e-05`、テスト所要時間7.39秒だった。
これは学習経路とマスクの正常性を確認する過学習試験であり、未知質問への一般化性能を
示すものではない。

再実行:

```sh
cmake --build build/release --target response_training_check -j2
build/release/response_training_check
```

