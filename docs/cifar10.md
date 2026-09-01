# cifar10.h 設計仕様

対象: `include/dataset_cifar10.h`

## 読み込む形式

公式CIFAR-10 binary版を対象とする。各レコードは3073 byteで、先頭1 byteが0〜9のlabel、続く3072 byteがR、G、Bそれぞれ1024 byteのplanar画像である。空ファイル、端数レコード、無効labelは例外となる。ファイルごとのレコード数は固定しない。

- `loadTraining(directory)`: `data_batch_1.bin`〜`data_batch_5.bin`を番号順に結合する。
- `loadTest(directory)`: `test_batch.bin`を読み込む。

## makeBatch

`makeBatch(indices, begin, count)`は指定順序のサンプルを入力3072 × countとtarget 10 × countへ変換する。空batch、indices範囲外、データセット範囲外のindexは例外となる。

RGB-planar入力は`(y * 32 + x) * 3 + channel`のHWC row順へ並べ替える。各pixelを255で割った後、チャネルごとに次で標準化する。

| channel | mean | std |
| --- | ---: | ---: |
| R | 0.4914 | 0.2470 |
| G | 0.4822 | 0.2435 |
| B | 0.4465 | 0.2616 |

targetは10クラスのone-hot表現で、戻り値の`nSize`は`count`と等しい。
