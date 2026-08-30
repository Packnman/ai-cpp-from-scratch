# cuda_function.h 設計仕様

対象: `lib/include/cuda_function.h`

## 目的

Tensor演算のforward/backwardインターフェースと、自動微分グラフへ演算を記録するContextを定義する。具体的なニューラルネットワーク演算も`Function`派生として提供する。

## 型別名

| 型 | 内容 |
| --- | --- |
| `TensorPtr` | `std::shared_ptr<Tensor>` |
| `TensorList` | TensorPtrの配列。Functionの複数入力・複数出力を表す |
| `TensorGradList` | 出力勾配cuMatへのconst pointer配列 |

## Function

### 派生クラス契約

- `forward(inputs)`: 数値計算を行い、1個以上の非null Tensorを返す。
- `backward(outputGrads, inputs, outputs)`: 上流勾配から入力やParameterの勾配を計算し、既存勾配へ加算する。
- Functionオブジェクトは、そのFunctionを記録した計算グラフのbackward完了まで生存しなければならない。

### apply

1. 派生クラスの`forward`を呼ぶ。
2. Function pointerと入力Tensorを保持するContextを生成する。
3. 全出力をContextへweak pointerで登録する。
4. 各出力Tensorへ同じContextを設定する。
5. 出力一覧を返す。

これにより出力Tensorから入力側へ計算グラフをたどれる。出力が空またはnullを含む場合は例外となる。

### operator()

単一出力Function向けの簡略API。`apply`を呼び、出力が正確に1個であることを検証して先頭Tensorを返す。複数出力Functionは`apply`を直接使う。

`forward`を直接呼ぶとContextが生成されないため、その演算は自動微分グラフへ記録されない。

## Context

| メンバー | 所有関係 | 目的 |
| --- | --- | --- |
| `_lpFunc` | 非所有raw pointer | backwardを呼ぶFunction |
| `_spmInputs` | shared ownership | backwardまで入力Tensorを保持 |
| `_wpmOutputs` | weak ownership | 出力との循環参照を防止 |

複数出力は1つのContextを共有する。Tensor側のグラフ探索はContext単位で重複を除外し、同一演算のbackwardを二重実行しない。

## ReLU

入力1個、出力は同形状。forwardは`max(0, x)`。backwardは入力が正の要素にだけ上流勾配を加算する。

## Dropout

### 構築

`Dropout(dropProbability, seed)`

drop probabilityは0以上1未満。cuRAND generatorをインスタンスごとに所有し、seedで初期化する。

### forward/backward

学習時に呼び出すことを前提とするinverted dropout。

```text
mask = 0                if dropped
mask = 1 / (1 - p)      if kept
output = input * mask
inputGrad += outputGrad * mask
```

p=0は恒等変換。同じインスタンスは最新forwardのmaskだけを保持するため、複数の未完了グラフを同時に保持してはならない。評価時の迂回は呼び出し側Moduleが担当する。

## BatchNorm

### 外部所有状態

コンストラクタは次のTensorへの非所有pointerを受け取る。

| Tensor | 種別 | 形状 |
| --- | --- | --- |
| gamma | Parameter | features × 1 |
| beta | Parameter | features × 1 |
| running mean | Buffer | features × 1 |
| running variance | Buffer | features × 1 |

momentumは0〜1、epsilonは正数。既定値は0.1と1e-5。

### モードと一時状態

初期状態は学習モード。`setTraining`で切り替える。forwardごとに正規化値と逆標準偏差を保存し、backwardで再利用する。`_wasTraining`により、forward時点のモードに対応する勾配式を選ぶ。

入力と出力はfeatures × batch。学習時はバッチ統計を使ってrunning統計を更新し、評価時はrunning統計を使う。backwardはinput、gamma、betaの勾配へ加算する。

同じインスタンスは最新forwardの一時状態だけを保持するため、複数の未完了グラフを同時に保持してはならない。詳細な数式はルートREADMEの「BatchNorm」を参照する。

## Linear

weightはoutputFeatures × inputFeatures、biasはoutputFeatures × 1。forwardはbatch列へbiasをbroadcastして次を計算する。

```text
Y = W X + b
```

backwardは次を加算する。

```text
inputGrad  += transpose(W) * outputGrad
weightGrad += outputGrad * transpose(X)
biasGrad   += outputGrad * ones
```

`_mTmp`は1 × batchのonesを再利用する作業領域。weightとbiasの所有権は呼び出し側Moduleにある。

## GELU

tanh近似によるGELUを同形状Tensorへ適用する。backwardは近似式の解析導関数を使って入力勾配へ加算する。

## SoftmaxCrossEntropy

入力は同形状のlogitsとone-hot targetの2個。形状はclasses × batch。forwardはバッチ平均cross entropyを1 × 1 Tensorで返す。

backwardはlogits勾配へ`upstream * (softmax - target) / batch`を加算する。target勾配は計算しない。

## ライフタイム上の制約

Linear、Dropout、BatchNormのようにraw pointerや一時bufferを持つFunctionは、通常Moduleのメンバーとして長期間保持する。ローカルFunctionを使う場合は、その出力に対するbackwardが終わる前にFunctionを破棄してはならない。
