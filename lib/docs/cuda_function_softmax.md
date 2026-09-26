# Softmax 設計仕様

対象: `lib/include/cuda_function_Softmax.h`

## 実装状態

指定軸のsliceごとにreduceする専用CUDA kernelとして実装済み。任意rankを処理し、数値安定なforwardとJacobian-vector積によるbackwardをGPU上で行う。Tensorデータはhostへ転送せず、入力検証用の小さなエラーフラグだけをhostへ戻す。

## 目的

指定軸の値を、合計1の非負な確率へ変換する。Multi-Head Attentionではmaskおよびscale適用後のscoreをkey軸に沿って正規化する。

## API

```cpp
explicit Softmax( std::size_t nAxis );
```

軸は構築時に保持し、forward時に入力rank未満であることを検証する。Transformerのscore shapeが `[queryLength, keyLength, heads, batch]` の場合、key軸は1なので `Softmax(1)` とする。

## Forward

入力 $X$ の指定軸に沿う各sliceについて、最大値を引く数値安定な式を使う。

$$
m=\max_j x_j
$$

$$
y_i=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)}
$$

出力shapeは入力shapeと同じであり、各sliceについて次を満たす。

$$
0\le y_i\le1,\qquad \sum_i y_i=1
$$

reduce処理は指定軸以外の各位置を独立に扱う。入力は変更せず、新しい連続Tensorへ出力する。

## Maskとの連携

`Mask` が無効位置を $-\infty$ にした場合、少なくとも1個の有限値を含むsliceではその位置のSoftmax出力は0になる。全要素が $-\infty$ のsliceは `-\infty-(-\infty)` により未定義となるため、Mask側で禁止し、Softmax側でも可能なら検出して例外にする。

v1ではNaNを含む入力を拒否する。$+\infty$ の扱いを実装する場合は、$+\infty$ の位置だけへ均等に確率を割り当てるなど、明示した規約とテストが必要である。単純実装では非有限値のうち、Mask由来の $-\infty$ だけを許可する。

## Backward

上流勾配を $G$、forward出力を $Y$ とすると、指定軸の各sliceについてJacobian-vector積を計算する。

$$
\frac{\partial L}{\partial X_i}
=
Y_i\left(
G_i-\sum_jG_jY_j
\right)
$$

完全なJacobian行列は生成しない。forward出力は `c_spmOutputs` から取得し、入力の既存勾配へ加算する。

## 検証と例外

- 入力数が1でなければ `std::runtime_error`
- 入力rankが0、またはaxisがrank以上なら `std::invalid_argument`
- 正規化軸のextentが0なら `std::invalid_argument`
- NaNまたは許可していない非有限値があれば `std::invalid_argument`
- backwardの出力または勾配shapeが入力shapeと異なれば `std::invalid_argument`
- shape積やkernel index上限を超える場合は `std::overflow_error`

## 完了条件

- 指定軸ごとの出力合計が許容誤差内で1になる
- 大きな正負の有限入力でもoverflowやunderflow由来のNaNを生じない
- Mask由来の $-\infty$ 位置が確率0になる
- backwardが数値微分と一致し、既存勾配へ加算される
- rank 2、3、4とaxisの違いをテストする
- 不正axis、空の正規化軸、全mask sliceを拒否する
