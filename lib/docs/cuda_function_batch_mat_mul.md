# BatchMatMul 設計仕様

対象: `lib/include/cuda_function_BatchMatMul.h`

## 実装状態

末尾バッチ軸を持つrow-major配置を直接扱う専用CUDA kernelとして実装済み。forward、Aの勾配、Bの勾配はすべてGPU上で計算し、hostとの間でTensorデータを転送しない。

## 目的

複数の行列積をバッチ単位で計算する。Multi-Head AttentionではQueryとKeyのスコア計算、およびattention weightとValueの積に使用する。

## shape規約

既存のTransformer用Tensorに合わせ、行列の2軸を先頭に置き、それ以降をバッチ軸とする。

```text
A: [M, K, B1, ..., Bn]
B: [K, N, B1, ..., Bn]
Y: [M, N, B1, ..., Bn]
```

rank 2の場合は通常の行列積であり、バッチ軸はない。v1では左右のバッチshapeが完全一致することを要求し、broadcastは行わない。転置指定もv1の公開APIには含めない。

各バッチ位置 $p=(b_1,\ldots,b_n)$ について次を計算する。

$$
Y_{i,j,p}=\sum_{k=0}^{K-1}A_{i,k,p}B_{k,j,p}
$$

## API契約

| メソッド | 入力 | 戻り値・効果 |
| --- | --- | --- |
| `forward(inputs)` | A、Bの順でTensor 2個 | shape `[M,N,B...]` のTensor 1個 |
| `backward(outputGrads, inputs, outputs)` | dYとforward時のA、B | A、Bの既存勾配へ加算 |

## Forward

1. 入力数、rank、内積軸 $K$、バッチshapeを検証する。
2. 出力shapeを計算して連続Tensorを確保する。
3. バッチごとに行列積を実行する。

このプロジェクトのrow-major配置では、末尾バッチ軸を持つ行列要素がメモリ上で交互に並ぶ。実装は次のいずれかとする。

- このlayoutを直接扱うCUDA kernelを実装する
- cuBLAS向けlayoutへ一時的にpermuteして連続化し、batched GEMM後に戻す

暗黙に通常のstrided batched GEMMへ渡し、誤ったstrideで計算してはならない。

## Backward

各バッチについて次を計算し、既存勾配へ加算する。

$$
\frac{\partial L}{\partial A}=G B^T
$$

$$
\frac{\partial L}{\partial B}=A^T G
$$

ここで $G=\partial L/\partial Y$ である。forward入力はContextが保持するため、backwardは保存されたAとBを利用する。

## 検証と例外

- 入力数が2でなければ `std::runtime_error`
- いずれかのrankが2未満なら `std::invalid_argument`
- $K$ が一致しなければ `std::invalid_argument`
- バッチshapeが一致しなければ `std::invalid_argument`
- shape積やCUDAのindex型を超える場合は `std::overflow_error`
- backwardの勾配shapeが出力shapeと異なる場合は `std::invalid_argument`

## 完了条件

- rank 2の積が `cuda_gemm` と一致する
- rank 3以上で全バッチの結果がCPU参照実装と一致する
- 非正方行列を処理できる
- AとBの勾配が数値微分と一致し、既存勾配へ加算される
- 内積軸またはバッチshapeの不一致を拒否する
