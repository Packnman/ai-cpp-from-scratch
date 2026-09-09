# LayerNorm 設計仕様

対象: `lib/include/cuda_function_LayerNorm.h`

## 実装状態

ソース基盤のみ存在し、現在の `forward` と `backward` は `std::logic_error` を送出する。現行コンストラクタには必要な設定がないため、実装時にAPIを拡張する。

## 目的

各系列位置・各バッチを独立に、特徴量軸で正規化する。Transformerのattentionブロックおよびfeed-forwardブロックの正規化に使用する。

## 計画API

実装時には少なくとも次をコンストラクタへ追加する。

```cpp
LayerNorm( Tensor* lpGamma, Tensor* lpBeta, float fEpsilon = 1.0e-5f );
```

`gamma` と `beta` は呼び出し側Moduleが所有するParameterであり、LayerNormは非所有pointerを保持する。FunctionとParameterは計算グラフのbackward完了まで生存しなければならない。

## shape規約

入力shapeを次とする。

```text
X:     [F, D1, ..., Dn]
gamma: [F, 1]
beta:  [F, 1]
Y:     [F, D1, ..., Dn]
```

$F$ は特徴量数であり、後続軸の各位置を独立した正規化単位とする。Transformerでは代表的に `[embeddingSize, sequenceLength, batch]` を入力する。

## Forward

後続軸の位置を $p$ として、特徴量軸の平均と母分散を計算する。

$$
\mu_p=\frac{1}{F}\sum_{f=0}^{F-1}x_{f,p}
$$

$$
\sigma_p^2=\frac{1}{F}\sum_{f=0}^{F-1}(x_{f,p}-\mu_p)^2
$$

$$
\hat{x}_{f,p}=\frac{x_{f,p}-\mu_p}{\sqrt{\sigma_p^2+\epsilon}}
$$

$$
y_{f,p}=\gamma_f\hat{x}_{f,p}+\beta_f
$$

分散計算は数値安定性を考慮し、極端な入力で負の分散やNaNを生じにくい方法を使う。

## Backward

上流勾配を $g_{f,p}$ とするとParameter勾配は次のとおりである。

$$
\frac{\partial L}{\partial\beta_f}=\sum_p g_{f,p}
$$

$$
\frac{\partial L}{\partial\gamma_f}=\sum_p g_{f,p}\hat{x}_{f,p}
$$

入力勾配は各位置 $p$ ごとに次を計算する。

$$
\frac{\partial L}{\partial x_{f,p}}=
\frac{\gamma_f}{F\sqrt{\sigma_p^2+\epsilon}}
\left[
Fg_{f,p}-\sum_jg_{j,p}
-\hat{x}_{f,p}\sum_jg_{j,p}\hat{x}_{j,p}
\right]
$$

すべての勾配は既存値へ加算する。v1ではbackward時に入力から平均・分散を再計算してよい。中間統計をインスタンスメンバーへ保存する場合は、同じFunctionを複数の未完了グラフで使えない制約を明記する。

## 検証と例外

- 入力は正確に1個
- 入力rankは1以上、特徴量数 $F$ は正
- gamma、betaは非nullかつ `[F,1]`
- epsilonは有限かつ正
- 出力勾配shapeは入力shapeと一致
- shape積が実装上限を超える場合は `std::overflow_error`

## 完了条件

- 各系列・バッチが独立に平均0、分散約1へ正規化される
- gammaとbetaのbroadcastが正しい
- input、gamma、betaの勾配が数値微分と一致する
- バッチサイズや系列長が変わっても同じインスタンスを利用できる
- 不正shape、null Parameter、不正epsilonを拒否する
