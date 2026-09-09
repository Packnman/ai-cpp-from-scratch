# Add 設計仕様

対象: `lib/include/cuda_function_Add.h`

## 実装状態

インターフェースとソース基盤のみ存在する。現在の `forward` と `backward` は `std::logic_error` を送出する。

## 目的

2個のTensorを要素単位で加算する。Transformerでは残差接続など、同じ形状の値を合流させる用途で使用する。

## API契約

`Add` は状態を持たない `Function` とする。

| メソッド | 入力 | 戻り値・効果 |
| --- | --- | --- |
| `forward(inputs)` | Tensor 2個 | 同じshapeのTensor 1個 |
| `backward(outputGrads, inputs, outputs)` | 出力勾配1個とforward時の入出力 | 両入力の既存勾配へ加算 |

v1ではbroadcastを行わず、2入力のshapeが完全一致することを要求する。出力shapeも入力shapeと同じである。

## Forward

入力を $A$、$B$、出力を $Y$ とすると、全要素について次を計算する。

$$
Y_i=A_i+B_i
$$

出力は新しい連続Tensorとして確保し、入力をin-place更新しない。CUDA側では既存の `cuda_geam`、または同等の要素単位kernelを利用できる。

## Backward

上流勾配を $G=\partial L/\partial Y$ とすると、両入力の勾配は同じである。

$$
\frac{\partial L}{\partial A}=G,\qquad
\frac{\partial L}{\partial B}=G
$$

autogradの共通規約に従い、代入ではなく次のように既存勾配へ加算する。

```text
A.grad += G
B.grad += G
```

同じTensorが2入力に指定された場合も2経路分を加算し、結果は `A.grad += 2 * G` となる。

## 検証と例外

- 入力数が2でなければ `std::runtime_error`
- null入力があれば `std::invalid_argument`
- 2入力のshapeが異なれば `std::invalid_argument`
- backwardの出力勾配が1個でない、またはnullなら `std::runtime_error`
- CUDA kernelの起動失敗は `std::runtime_error`

## 完了条件

- rank 2だけでなく任意rankの同形状Tensorを処理できる
- forwardの数値結果がCPU参照実装と一致する
- backwardが両入力へ加算される
- 同一Tensorを2回渡した場合に勾配が2倍になる
- shape不一致と入力数不一致を拒否する
