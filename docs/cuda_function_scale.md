# Scale 設計仕様

対象: `lib/include/cuda_function_Scale.h`

## 実装状態

ソース基盤のみ存在し、現在の `forward` と `backward` は `std::logic_error` を送出する。現行コンストラクタには倍率の指定がないため、実装時にAPIを拡張する。

## 目的

Tensorの全要素へ固定scalarを乗算する。Scaled Dot-Product AttentionではQueryとKeyの内積を $1/\sqrt{d_k}$ 倍し、Softmaxが飽和しにくい範囲へ調整する。

## 計画API

```cpp
explicit Scale( float fScale );
```

倍率は学習ParameterではなくFunctionの固定設定とする。したがって倍率自身の勾配は計算しない。

Multi-Head Attentionでの代表的な生成例は次のとおりである。

```cpp
Scale scale( 1.0f / std::sqrt( static_cast<float>(nHeadSize) ) );
```

## API契約

| メソッド | 入力 | 戻り値・効果 |
| --- | --- | --- |
| `forward(inputs)` | 任意rankのTensor 1個 | 入力と同shapeのTensor 1個 |
| `backward(outputGrads, inputs, outputs)` | 出力勾配1個 | 入力の既存勾配へ加算 |

## Forward

倍率を $s$ とすると全要素について次を計算する。

$$
Y_i=sX_i
$$

出力は新しい連続Tensorとし、入力をin-place更新しない。実装には既存の `cuda_scale` を使用できるが、出力へ入力dataをdevice-to-device copyしてからscaleする必要がある。専用kernelで直接別bufferへ書き込んでもよい。

## Backward

上流勾配を $G$ とすると次を入力の既存勾配へ加算する。

$$
\frac{\partial L}{\partial X_i}=sG_i
$$

`cuda_axpy(inputGrad, s, outputGrad)` を利用できる。

## 特殊値

- `s == 0` は許可し、出力と入力勾配は0になる
- 負の倍率を許可する
- NaNまたは正負の無限大は設定ミスとして拒否する

## 検証と例外

- コンストラクタで倍率が有限でなければ `std::invalid_argument`
- 入力数が1でなければ `std::runtime_error`
- backwardの出力勾配がnull、またはshape不一致なら `std::runtime_error` または `std::invalid_argument`
- CUDA演算失敗は `std::runtime_error`

## 完了条件

- 正、負、0の倍率でCPU参照結果と一致する
- 任意rankでshapeを保持する
- backwardが倍率を反映して既存勾配へ加算される
- 非有限倍率、入力数不一致、勾配shape不一致を拒否する
