# Permute 設計仕様

対象: `lib/include/cuda_function_Permute.h`

## 実装状態

任意rankの軸順序を直接変換する専用CUDA kernelとして実装済み。forwardでは指定順列、backwardでは逆順列をGPU上で適用し、引数なしの構築は恒等順列として扱う。

## 目的

Tensorの要素値を変えずに軸順序を入れ替える。Transformerではhead軸の分割・結合や、BatchMatMulへ渡すQuery、Key、Valueの軸調整に使用する。

## API

出力軸から入力軸への対応をコンストラクタで受け取る。

```cpp
explicit Permute( std::vector<std::size_t> dimensions );
```

`dimensions[i]` は「出力の軸 `i` が入力のどの軸に対応するか」を表す。

例:

```text
input shape:      [2, 3, 4]
dimensions:       [1, 0, 2]
output shape:     [3, 2, 4]
output[j,i,k] = input[i,j,k]
```

コンストラクタは順方向の軸順序と、その逆順序を保持する。

## Forward

1. 入力が1個であることを検証する。
2. `dimensions` が入力rankと同じ長さの完全な順列であることを検証する。
3. 出力shapeと入力strideをCUDA kernelへ渡す。
4. 各出力要素の座標から入力offsetを求め、新しい連続TensorへGPU上で書き込む。

Tensorはdataとgradを同shapeで所有し、多くのCUDA演算は連続配置を要求する。このためv1のFunction出力は非連続viewではなく、連続した独立Tensorとする。

## Backward

出力勾配へ逆順列を適用し、入力shapeへ戻してから既存の入力勾配へ加算する。

順列を $P$、逆順列を $P^{-1}$ とすると次の関係になる。

$$
Y=P(X),\qquad
\frac{\partial L}{\partial X}=P^{-1}\left(\frac{\partial L}{\partial Y}\right)
$$

逆順列を同じCUDA kernelへ渡して連続Tensorへ書き込み、入力の既存勾配へGPU上で加算する。

## 検証と例外

- 入力数が1でなければ `std::runtime_error`
- 軸指定数と入力rankが異なれば `std::invalid_argument`
- 軸番号がrank以上、または重複していれば `std::invalid_argument`
- backwardの出力勾配shapeがforward出力shapeと異なれば `std::invalid_argument`
- CUDA copyまたはkernel起動に失敗した場合は `std::runtime_error`

## 完了条件

- rank 2、3、4の代表的な順列がCPU参照結果と一致する
- 恒等順列でも正しい同shape出力を返す
- backwardが逆順列となり、既存勾配へ加算される
- 不正な軸、重複軸、rank不一致を拒否する
- 出力dataとgradが連続配置になる
