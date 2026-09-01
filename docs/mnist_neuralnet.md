# neuralnet.h 設計仕様

対象: `include/neuralnet_mnist.h`

## 全体構成

MNIST分類器を次の順序で構成する。

```text
784入力
  → Hidden(784, 256)
  → Hidden(256, 128)
  → Linear(128, 10)
  → logits
```

各Hidden層の内部順序は次の通り。

```text
Linear → BatchNorm → ReLU → Dropout
```

## LayerInput

入力前処理用の拡張点。現在の`forward`は入力Tensorをそのまま返し、パラメータや状態を持たない。`init`も処理を行わない。

呼び出し側は1個以上の入力を渡す必要がある。現実装は先頭要素へ直接アクセスするため、空入力の検証は上位の`MnistNeuralNet`が担う。

## LayerHidden

### 所有状態

| 種別 | 名前 | 形状 |
| --- | --- | --- |
| Parameter | weight | nOutput × nInput |
| Parameter | bias | nOutput × 1 |
| Parameter | gamma | nOutput × 1 |
| Parameter | beta | nOutput × 1 |
| Buffer | running_mean | nOutput × 1 |
| Buffer | running_var | nOutput × 1 |

ParameterはOptimizerの更新対象、Bufferは保存対象だが更新対象外である。

### 初期化

- weight: 平均0、標準偏差 `sqrt(2 / fan_in)` の正規分布
- bias: 0
- gamma: 1
- beta: 0
- running_mean: 0
- running_var: 1

### モード

`Module::isTraining()`をBatchNormへ毎forward同期する。学習時はDropoutを適用し、評価時はDropoutを迂回する。BatchNormは学習時にバッチ統計とrunning統計を使い、評価時にrunning統計を使う。

## LayerOutput

weightとbiasを所有・登録し、Linearだけを適用する。初期化はHidden層のLinear部分と同じである。BatchNorm、活性化関数、Dropoutは適用しない。

## MnistNeuralNet

### コンストラクタ

`MnistNeuralNet(seed, dropoutRate = 0.2f)`

モデル初期化用乱数を`seed`で初期化する。2個のDropoutにはそれぞれ`seed + 1`と`seed + 2`を渡し、異なる乱数系列を使用する。

子Moduleは`input`、`hidden1`、`hidden2`、`output`の名前で登録する。

### forward

入力は非nullなTensor 1個で、形状は784 × batch、batchは1以上でなければならない。戻り値は10 × batchのlogitsである。

### loss

入力と10 × batchのone-hot targetから`SoftmaxCrossEntropy`を計算し、1 × 1の損失Tensorを返す。

### 永続化される状態

パラメータは各Hiddenのweight、bias、gamma、betaとOutputのweight、bias。Bufferは各Hiddenのrunning_meanとrunning_varである。Dropout mask、BatchNormの一時正規化値、Optimizer状態はモデル状態へ含めない。
