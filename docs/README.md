# 設計仕様書

このディレクトリには、公開ヘッダーごとの設計仕様をまとめる。仕様は現在の宣言と実装を基準とし、責務、所有権、データ形状、API契約、例外条件、利用上の注意を記載する。

## アプリケーション層

| ヘッダー | 設計仕様 |
| --- | --- |
| `include/mnist.h` | [mnist.md](mnist.md) |
| `include/neuralnet.h` | [neuralnet.md](neuralnet.md) |
| `include/trainer.h` | [trainer.md](trainer.md) |

## ライブラリ層

| ヘッダー | 設計仕様 |
| --- | --- |
| `lib/include/cuda_bublas.h` | [cuda_bublas.md](cuda_bublas.md) |
| `lib/include/cuda_function.h` | [cuda_function.md](cuda_function.md) |
| `lib/include/cuda_matrix.h` | [cuda_matrix.md](cuda_matrix.md) |
| `lib/include/cuda_tensor.h` | [cuda_tensor.md](cuda_tensor.md) |
| `lib/include/matrix.h` | [matrix.md](matrix.md) |
| `lib/include/module.h` | [module.md](module.md) |
| `lib/include/optimizer.h` | [optimizer.md](optimizer.md) |

## 共通規約

- ニューラルネットワークの行列は「特徴量またはクラス数 × バッチ数」で表す。
- `Mat`と`cuMat`はcolumn-majorであり、要素位置は `column * rows + row` である。
- `Tensor`のデータと勾配はGPU上に置く。
- 学習パラメータとBufferの所有権は派生`Module`が持ち、基底`Module`は非所有ポインタを登録する。
- CUDA演算のbackwardは、原則として既存の勾配へ加算する。
