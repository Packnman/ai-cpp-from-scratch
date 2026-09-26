# cuda_matrix.h 設計仕様

## cuStorage / cuMat

`cuStorage<T>`はCUDA allocationを所有し、`cuMat<T>` viewから`shared_ptr`で共有される。
C++20の`cuElement` conceptにより、要素型は`float`と`std::int32_t`だけに制限される。
公開aliasはfloat32用の`cufStorage` / `cufMat`と、token ID用int32の
`cunStorage` / `cunMat`である。どちらもN次元shape、row-major stride、
storage offsetを持ち、新規確保は常にcontiguousである。

```text
2D index(row, col) = row * cols + col
```

`reshape`はcontiguous tensorの要素数を変えずにviewを返す。`permute`と`slice`もstorageを共有するviewであり、`contiguous`でrow-major allocationへ実体化できる。viewが元tensorより長く生存してもshared storageは保持される。

`copyFromHost`、`copyToHost`、`toHost`は両要素型でrow-major順を保ち、
転置を行わない。`download(Mat)`はhostからdevice、`upload(Mat)`はdeviceからhostで、
`ones()`および下記CUDA演算とともに`cufMat`だけで利用できる。

## 演算

flat演算は`numel()`を使用する。行列演算はcontiguous rank-2 tensorを要求する。

`cuda_gemm`の公開規約はrow-majorの`C = op(A) * op(B)`である。cuBLASには同じmemoryをcolumn-majorの転置行列と見せ、operandを交換した`C^T = op(B)^T * op(A)^T`として渡すため、転置copyは発生しない。A/Bのtranspose全4組合せを扱う。

ニューラルネットワークの2次元論理shapeは引き続き`[features,batch]`であり、アドレスは`feature * batch + sample`となる。BatchNorm、SoftmaxCrossEntropy、Conv2D、Poolingもこの規約を使用する。
