# neuralnet.h 設計仕様

対象: `include/neuralnet_cifar10.h`

## Cifar10NeuralNet

CIFAR-10分類器を次の順序で構成する。

```text
3072 (32×32×3 HWC)
  → Conv2D(3→32, 3×3, stride=1, padding=1)
  → ReLU → MaxPool2D(2×2, stride=2)
  → Conv2D(32→64, 3×3, stride=1, padding=1)
  → ReLU → MaxPool2D(2×2, stride=2)
  → 4096 → Linear(256) → BatchNorm → ReLU → Dropout
  → Linear(10) → logits
```

`Cifar10NeuralNet(seed, dropoutRate = 0.2f)`は畳み込み層と全結合層のweightをHe初期化する。Dropoutには`seed + 1`を使う。出力へSoftmaxは適用せず、`loss`がlogitsとone-hot targetを`SoftmaxCrossEntropy`へ渡す。

### 入出力契約

`forward`は非nullなTensor 1個を受け取る。入力形状は3072 × batch、batchは1以上であり、戻り値は10 × batchのlogitsである。`loss`のtargetは10 × batchでなければならず、戻り値は1 × 1である。不正な個数、null、shapeは`std::runtime_error`となる。

画像rowは`(y * 32 + x) * 3 + channel`のHWC順である。各畳み込みブロックも同じHWC順を保ち、1段目は8192 × batch、2段目は4096 × batchを返す。

### モードと状態

`setTraining(false)`は子Moduleへ伝播する。Dense層は評価時にBatchNormのrunning統計を使い、Dropoutを迂回する。

Parameter名は`conv1.conv.*`、`conv2.conv.*`、`hidden.*`、`output.*`である。Dense層の`running_mean`と`running_var`はBufferとして保存される。
