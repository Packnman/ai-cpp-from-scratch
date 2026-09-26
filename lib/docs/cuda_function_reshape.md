# Reshape 設計仕様

対象: `lib/include/cuda_function_Reshape.h`

## 実装状態

実装済み。要素数を維持したshape変更と1軸の自動推論を行い、backwardでは元のshapeへ戻して勾配を加算する。引数なしの構築は恒等reshapeとして扱う。

## 目的

row-majorの線形要素順序を変えずにTensorのshapeだけを変更する。Transformerではembedding軸をhead数とhead次元へ分割する処理、および分割したheadを再結合する処理に使用する。

## API

```cpp
explicit Reshape( std::vector<std::int64_t> shape );
```

target shapeには最大1個の `-1` を許可し、入力要素数から自動推論する。それ以外の負値は不正とする。推論後のtarget要素数は入力要素数と完全一致しなければならない。

例:

```text
input:  [embeddingSize, sequence, batch]
target: [headSize, heads, sequence, batch]

headSize * heads == embeddingSize
```

## Forward

1. 入力数と入力の連続性を検証する。
2. `-1` があれば具体的なextentへ解決する。
3. 入出力の要素数が同じであることを検証する。
4. row-majorの線形順序を保持してtarget shapeのTensorを返す。

現行Tensorには既存 `cufMat` viewを受け取るコンストラクタがない。v1ではtarget shapeのTensorを新規確保してdevice-to-device copyする。将来Tensorが安全なview所有をサポートした場合は、コピーなしreshapeへ変更できる。

## Backward

上流勾配の線形要素順序を変えずに元の入力shapeへ戻し、入力の既存勾配へ加算する。

$$
Y=\operatorname{reshape}(X),\qquad
\operatorname{vec}(Y)=\operatorname{vec}(X)
$$

$$
X.grad\mathrel{+}=\operatorname{reshape}(Y.grad,\operatorname{shape}(X))
$$

## zero extentと`-1`

- 明示的な0 extentは基盤 `cufMat` と同様に許可する
- 入力要素数が0でtargetに `-1` がある場合、推論が一意にならないため拒否する
- `-1` がない場合は、target要素数が0で入力要素数も0なら許可する

## 検証と例外

- 入力数が1でなければ `std::runtime_error`
- 非連続入力はv1では `std::invalid_argument`
- `-1` が2個以上、または `-1` 未満のextentがあれば `std::invalid_argument`
- 要素数が一致しなければ `std::invalid_argument`
- shape積が `std::size_t` を超える場合は `std::overflow_error`
- backwardの勾配shapeが出力shapeと異なれば `std::invalid_argument`

## 完了条件

- rankの増減と恒等reshapeを処理できる
- `-1` を正しく推論できる
- forward/backwardで線形要素順序が保持される
- backwardが既存勾配へ加算される
- 要素数不一致、複数の `-1`、非連続入力を拒否する
