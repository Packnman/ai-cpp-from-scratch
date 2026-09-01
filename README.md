# ai-cpp-from-scratch

C++20とCUDAでニューラルネットワークの基盤を低レベルから実装し、MNISTとCIFAR-10の分類モデルを学習・評価するプロジェクトです。CPU/GPUの行列演算、自動微分、モデルの状態管理、Optimizer、データセット処理、学習ループまでをリポジトリ内で実装しています。

公開APIの契約、データ形状、所有権、数式などの詳細は[設計仕様書](docs/README.md)を参照してください。

## 実装済みの主な機能

- CPU行列`Mat`とCUDAデバイス行列`cuMat`
- cuBLASとCUDA kernelによる行列・ニューラルネットワーク演算
- `Tensor`、`Function`、`Context`による自動微分
- Linear、Conv2D、MaxPool2D、BatchNorm、Dropout、ReLU、GELU、Softmax Cross Entropy
- SGD、Adam
- `Module` / `Model`によるParameter、Buffer、子Module、学習・評価モード、モデル保存・読込の管理
- DatasetとModelを差し替えられる教師あり学習・評価用trainer
- MNIST IDXファイルの読込、バッチ生成、分類精度の集計

- CIFAR-10 binary batchの読込、HWC変換、チャネル標準化
## MNISTモデル

現在の分類モデルは次の構成です。

```text
784 → 256 → 128 → 10
```

2つの隠れ層では次の順に演算します。出力層はLinearのみで、10クラスのlogitsを返します。

```text
Linear → BatchNorm → ReLU → Dropout
```

## CIFAR-10モデル

```text
32×32×3 HWC
→ Conv2D(3→32, 3×3, padding=1) → ReLU → MaxPool2D
→ Conv2D(32→64, 3×3, padding=1) → ReLU → MaxPool2D
→ 4096 → Linear(256) → BatchNorm → ReLU → Dropout
→ Linear(10)
```

最終層はSoftmaxを適用しない10クラスのlogitsを返し、学習時にSoftmax Cross Entropyへ渡します。

## 必要環境

- CMake 3.18以降
- C++20対応コンパイラ
- NVIDIA GPU
- CUDA Toolkit（CUDA Runtime、cuBLAS、cuRANDを含む）

開発コンテナはCUDA 12.8ベースです。既定ではCompute Capability 8.6向けにビルドします。別のGPUを使用する場合は、configure時に`AI_CPP_CUDA_ARCHITECTURES`を変更してください。

```sh
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAI_CPP_CUDA_ARCHITECTURES=89
```

## データの準備

MNISTの未圧縮IDXファイルと、公式CIFAR-10 binary版を次のように配置します。Python pickle版には対応しません。モデル保存先の`models/`も事前に作成してください。

```text
data/
├── mnist/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
└── cifar-10-batches-bin/
    ├── data_batch_1.bin
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    └── test_batch.bin
models/
```

CIFAR-10は各レコードのRGB-planarデータをHWC順へ変換し、`pixel / 255`の後にmean `(0.4914, 0.4822, 0.4465)`、std `(0.2470, 0.2435, 0.2616)`でチャネル別に標準化します。

```sh
mkdir -p models
```

## ビルド

リポジトリルートで次を実行します。

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
```

主なビルドターゲットは次のとおりです。

| ターゲット | 役割 |
| --- | --- |
| `ai_cpp` | 行列、自動微分、Module、Optimizerを含む静的ライブラリ |
| `mnist_train` | MNISTを学習し、テストデータで評価してモデルを保存する実行ファイル |
| `cuda_check` | CUDAデバイス検出と基本的なkernel実行のテスト |
| `multi_output_check` | 複数出力を持つ自動微分グラフのテスト |
| `cifar10_train` | CIFAR-10を学習し、テストデータで評価してモデルを保存する実行ファイル |
| `model_state_check` | Parameter、Buffer、モード、モデル保存・読込のテスト |
| `dropout_check` | Dropoutの値検証、再現性、学習・評価モードのテスト |
| `batchnorm_check` | BatchNormのforward、backward、running統計のテスト |
| `conv2d_check` | Conv2D、LayerConv2D、勾配、shape、初期化のテスト |
| `maxpool2d_check` | MaxPool2Dのforward、backward、tie、重複領域のテスト |

## テスト

| `cifar10_dataset_check` | CIFAR-10 binary読込、変換、検証のテスト |
| `cifar10_model_check` | CIFAR-10モデルのshape、backward、モード、保存・読込のテスト |
CTestには9件のテストが登録されています。

```sh
ctest --test-dir build/debug --output-on-failure
```

`cuda_check`だけをconfigure、build、実行する場合は次のスクリプトを使用できます。

```sh
./scripts/test_cuda_check.sh
```

## MNISTの学習と評価

ビルド後、データディレクトリとモデル保存ディレクトリを指定して実行します。

```sh
./build/debug/mnist_train data/mnist models
```

現在の設定値はコードに固定されています。

| 設定 | 値 |
| --- | --- |
| epoch | 5 |
| batch size | 128 |
| seed | 42 |
| Optimizer | Adam |
| learning rate | `1e-6` |
| dropout probability | 0.2 |

各epochの平均lossとテスト精度を標準出力へ表示し、学習済みモデルを`models/mnist_model_batchnorm.bin`へ保存します。

データを準備した状態で、configure、全ターゲットのbuild、CTest、MNIST学習・評価をまとめて実行するには次を使用します。

```sh
./scripts/run.sh
```
## CIFAR-10の学習と評価

引数を省略すると`data/cifar-10-batches-bin`を読み、`models/cifar10_model.bin`へ保存します。

```sh
./build/debug/cifar10_train data/cifar-10-batches-bin models
```

既定設定は20 epoch、batch size 128、seed 42、Adam、learning rate `1e-3`、dropout probability 0.2です。各epochの平均lossとテスト精度を表示してからモデルを保存します。


## ディレクトリ構成

```text
.
├── docs/          公開ヘッダー別の設計仕様書
├── include/       MNIST・CIFAR-10アプリケーション層の公開ヘッダー
├── src/           MNIST・CIFAR-10アプリケーションと実行ファイル
├── lib/
│   ├── include/   行列・Tensor・Module・Optimizerの公開ヘッダー
│   └── src/       C++/CUDAライブラリ実装
├── tests/         CTestから実行するテストプログラム
├── scripts/       ビルド・テスト・学習用スクリプト
├── data/          MNIST・CIFAR-10データ（ローカル配置）
└── models/        学習済みモデルの保存先（ローカル配置）
```

## 現在の制約

画像演算のTensorは `(height * width * channels) x batch` の2次元で、rowは `(y * width + x) * channels + channel` のHWC順です。Conv2Dはkernelを反転しないcross-correlationで、正方形kernel、縦横共通stride、対称padding、必須biasに対応します。MaxPool2Dはpaddingなしで、最大値が同値ならHWC走査順の最初を選びます。

- NVIDIA GPUとCUDA環境が必須で、CPU-only実行には対応していません。
- epoch、batch size、Optimizer、学習率などを変更するCLIはなく、ハイパーパラメータは各実行ファイルに固定されています。
- `Model::save`が保存するのはParameterとBufferです。Optimizer状態とepoch・stepなどの学習進捗は保存されないため、中断位置からの再開学習には対応していません。
