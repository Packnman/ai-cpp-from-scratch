# mnist.h 設計仕様

対象: `include/dataset_mnist.h`

## 目的

MNISTのIDXファイルを読み込み、学習ループが要求する`SupervisedBatch`へ変換する。

## MnistDataset

### 責務

- 画像IDXファイルとラベルIDXファイルのロード
- CPUメモリ上での生データ保持
- 指定されたサンプル順序からGPU Tensorのバッチを構築

### 内部状態

| メンバー | 内容 |
| --- | --- |
| `_nImages` | 全画像のuint8ピクセル。1画像784要素で連続格納 |
| `_nLabels` | 各画像に対応する0〜9のラベル |

### load

`static MnistDataset load(imagePath, labelPath)`

IDXヘッダーをbig-endianとして読み取る。画像magicは2051、ラベルmagicは2049、画像サイズは28×28でなければならない。画像数は1以上かつラベル数と一致し、全ラベルは0〜9でなければならない。

ファイルを開けない、ヘッダー不正、データ欠損、サイズ不一致、無効なラベルの場合は`std::runtime_error`を送出する。

### size

ラベル数、すなわちデータセットのサンプル数を返す。

### makeBatch

`SupervisedBatch makeBatch(indices, begin, count) const`

`indices[begin, begin + count)`が指すサンプルを次のTensorへ変換する。

| Tensor | 形状 | 内容 |
| --- | --- | --- |
| input | 784 × count | ピクセル値を255で割った0〜1のfloat |
| target | 10 × count | ラベルのone-hot表現 |

戻り値の`nSize`は`count`と等しい。範囲が`indices`を超える場合は例外となる。呼び出し側は、各インデックスがデータセットサイズ未満であることと、`count`がTensorで表現可能な範囲であることを保証する。
