# optimizer.h 設計仕様

対象: `lib/include/optimizer.h`

## 目的

Modelから学習Parameterを取得し、Tensorに蓄積された勾配でParameterを更新する。Optimizer固有のParameter単位状態も管理する。

## OptimizerState

テンプレート構造体で、学習率、step、Parameter状態配列をまとめる。Adam向けには`AdamState = OptimizerState<NamedAdamState>`が定義される。

現在は状態を表現する型だけがあり、Optimizerからexport/importまたはファイル保存する公開APIは未実装。

## OptimizerParams

ParameterごとのOptimizer状態を表すpolymorphic基底。派生状態を`shared_ptr<OptimizerParams>`として保持するためvirtual destructorを持つ。

## Optimizer

### 所有権

- `_lpModel`: 非所有pointer。Optimizerより長く生存する必要がある。
- `_lpParams`: Modelが所有するTensorへの非所有pointer。
- `_spOptimizerParams`: Optimizerが所有するParameter単位状態。

null Modelはコンストラクタで拒否する。

### init

stepを0へ戻し、`Model::getParams`で現在のParameter一覧を取得する。各Parameterについて派生Optimizerの`createOptimizerParams`を呼び、状態を再構築する。

学習開始前に必ず呼ぶ。再度呼ぶとmomentumなどのOptimizer状態はリセットされる。ModelのParameter登録構成を変更した場合も再初期化が必要。

### update

stepを1増加させ、Parameterと対応状態を同じ登録順で`update_param`へ渡す。勾配は更新後も自動ではゼロにならない。

### zero_grads

Modelの`zero_grads`へ委譲し、全学習Parameterの勾配を0へ設定する。

## SGDParams / SGD

SGDParamsは追加状態を持たない。更新式は次の通り。

```text
parameter -= learningRate * gradient
```

weight decay、momentum、Nesterovは実装しない。

## NamedAdamState / AdamState

`NamedAdamState`は状態名と1次・2次モーメントcuMatへのpointerを表す。checkpoint向けの表現を意図しているが、現Optimizerとの変換APIは未実装。

## AdamParams

各Parameterと同形状の`_mM`と`_mV`を所有し、構築時に0初期化する。

## Adam

固定hyperparameterを使用する。

| 値 | 設定 |
| --- | --- |
| beta1 | 0.9 |
| beta2 | 0.999 |
| epsilon | 1e-8 |
| learning rate | コンストラクタ引数 |

stepごとにbias correctionを計算し、CUDA kernelで次をin-place更新する。

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
mHat = m / (1 - beta1^step)
vHat = v / (1 - beta2^step)
parameter -= learningRate * mHat / (sqrt(vHat) + epsilon)
```

`update_param`はParameter状態を`AdamParams`へdynamic castし、型不一致なら例外を送出する。

## 制約

- 学習率の正値検証は現在行わない。
- gradient clipping、weight decay、Parameter groupは未実装。
- Optimizer状態はModel::saveへ含まれない。
- Parameter順序はModuleの登録順に依存する。
