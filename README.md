# ai-cpp-from-scratch

CUDA と C++ を使い、ニューラルネットワークの基本要素を低レベルから実装する開発中のプロジェクトです。行列演算、Tensor の自動微分、活性化関数、損失関数、Optimizer など、学習基盤を構成する部品を実装しています。

Module と Model は派生クラスでネットワーク構造やパラメータを定義することを前提とした基盤です。エンドツーエンドで学習を実行するプログラムはまだ実装されておらず、このリポジトリは完成済みの学習アプリケーションではありません。

公開ヘッダーごとの設計仕様は [docs/README.md](docs/README.md) を参照してください。

## 実装済みの主な機能

- CPU の `Mat` と CUDA デバイス上の `cuMat` による行列データ管理
- cuBLAS および CUDA kernel を使った行列演算
- `Tensor`、`Function`、`Context` による逆伝播の基盤
- Linear、BatchNorm、Dropout、ReLU、GELU、Softmax Cross Entropy
- SGD、Adam
- Module/Model の保存、読み込み、勾配初期化のための基底 API
- CUDA デバイス検出と簡単な kernel 実行を確認する `cuda_check`

## 必要環境

- CMake 3.18 以降
- C++20 対応コンパイラ
- NVIDIA GPU
- CUDA Toolkit（CUDA Runtime と cuBLAS を含む）

現在の開発コンテナは CUDA 12.8、Compute Capability 8.6 を基準にしています。既定の CUDA architecture は `86` です。別の GPU 向けにビルドする場合は、configure 時に `AI_CPP_CUDA_ARCHITECTURES` を指定してください。

```sh
cmake -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DAI_CPP_CUDA_ARCHITECTURES=89
```

## ビルドと動作確認

リポジトリルートで次を実行します。

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
```

`ai_cpp` は全実装を含む静的ライブラリです。`cuda_check` は CUDA デバイスを検出し、簡単な kernel が実行できることを確認する CTest のテストターゲットです。

configure、全ターゲットのビルド、CTest は次のスクリプトでもまとめて実行できます。

```sh
./scripts/run.sh
```

CUDA 確認ターゲットだけをビルドしてテストする場合は次を実行します。

```sh
./scripts/test_cuda_check.sh
```

## ディレクトリ構成

```text
.
├── include/       MNISTアプリケーション用ヘッダー
├── src/           MNISTアプリケーション実装
├── lib/
│   ├── include/   行列・Tensor・Module・Optimizerの公開ヘッダー
│   └── src/       C++/CUDAライブラリ実装
├── tests/         CUDA動作確認プログラム
└── scripts/       ビルド・テスト用スクリプト
```

## 数式・実装メモ


### Linear
$$
\boldsymbol{X}=\boldsymbol{U}^{(1)}
\\
\boldsymbol{Z}^{(i)} = f^{(i)}( \boldsymbol{U^{(i)}})
\\
U^{(i+1)} = \boldsymbol{W}^{(i)}\boldsymbol{Z}^{(i)} + \boldsymbol{b}^{(i)}
\\
\Delta^{(i)} = (\boldsymbol{W^{(i+1)T}} \Delta^{(i+1)}) \odot f^{(i)'}( \boldsymbol{U}^{(1)})
\\
\frac{\partial E_p}{\partial b} = \Delta^{(i)}\boldsymbol{v}^T
$$

### BatchNorm

この実装では、入力を「特徴量数 × バッチ数」の行列として扱い、各特徴量をバッチ方向に正規化する。
$f$を特徴量の添字、$b$をバッチ内サンプルの添字、$B$をバッチ数とする。

| 記号 | 意味 |
| -- | :-- |
| $x_{fb}$ | BatchNormへの入力 |
| $\mu_f$ | ミニバッチ平均 |
| $\sigma_f^2$ | ミニバッチ分散 |
| $\hat{x}_{fb}$ | 正規化後の値 |
| $\gamma_f,\beta_f$ | 学習するスケールとシフト |
| $\epsilon$ | ゼロ除算を防ぐ微小値 |
| $m$ | running統計の更新率（momentum） |

既定値は $m=0.1$、$\epsilon=10^{-5}$ とする。

学習時の順伝播は次の通り。

$$
\mu_f=\frac{1}{B}\sum_{b=1}^{B}x_{fb}
\\
\sigma_f^2=\frac{1}{B}\sum_{b=1}^{B}(x_{fb}-\mu_f)^2
\\
\hat{x}_{fb}=
\frac{x_{fb}-\mu_f}{\sqrt{\sigma_f^2+\epsilon}}
\\
y_{fb}=\gamma_f\hat{x}_{fb}+\beta_f
$$

学習中は評価用の平均と分散も更新する。

$$
\mathrm{running\_mean}_f
\leftarrow
(1-m)\mathrm{running\_mean}_f+m\mu_f
\\
\mathrm{running\_var}_f
\leftarrow
(1-m)\mathrm{running\_var}_f
+m\frac{B}{B-1}\sigma_f^2
\qquad (B>1)
$$

正規化には母分散（分母$B$）を使い、running varianceの更新には不偏分散（分母$B-1$）を使う。$B=1$の場合は補正しない。

評価時はミニバッチの統計を計算・更新せず、保存済みのrunning統計を使用する。

$$
\hat{x}_{fb}=
\frac{x_{fb}-\mathrm{running\_mean}_f}
{\sqrt{\mathrm{running\_var}_f+\epsilon}}
\\
y_{fb}=\gamma_f\hat{x}_{fb}+\beta_f
$$

逆伝播では、上流勾配を

$$
g_{fb}=\frac{\partial L}{\partial y_{fb}}
$$

とすると、学習パラメータの勾配は次になる。

$$
\frac{\partial L}{\partial \beta_f}
=\sum_{b=1}^{B}g_{fb}
\\
\frac{\partial L}{\partial \gamma_f}
=\sum_{b=1}^{B}g_{fb}\hat{x}_{fb}
$$

学習時の入力勾配は次の式でまとめて計算する。

$$
\frac{\partial L}{\partial x_{fb}}
=
\frac{\gamma_f}
{B\sqrt{\sigma_f^2+\epsilon}}
\left[
Bg_{fb}
-\sum_{j=1}^{B}g_{fj}
-\hat{x}_{fb}
\sum_{j=1}^{B}g_{fj}\hat{x}_{fj}
\right]
$$

gammaとbetaはParameterとしてOptimizerの更新対象にし、running_meanとrunning_varはBufferとしてモデルへ保存する。現在の隠れ層では Linear → BatchNorm → ReLU → Dropout の順序で適用する。

### ReLU
$$
max(0,x)
$$

### GELU
順伝播
$$
x\Phi(x)
\\
\Phi(x) = \frac{1}{2}
\left(
1+erf\left(\frac{x}{\sqrt{2}}\right)
\right)
$$
\
近似式は
$$
\frac{1}{2}x
\left[
1+tanh
\left(
\sqrt{\frac{2}{\pi}}(x+0.0447115x^3)
\right)
\right]
$$
整理すると
$$
\\
y=\frac{1}{2}x(1+tanh(u))
\\
u=c(x+0.044715x^3)
$$
\
逆伝播
$$
\left(
x\Phi(x)
\right)'
=
\Phi(x) + x\phi(x)
\\
=\frac{1}{2}
\left(
1+erf\left(\frac{x}{\sqrt{2}}\right)
\right)
+
\frac{x}{\sqrt{2\pi}}e^{-x^2/2}
$$
近似式は、
$$
\frac{dy}{dx}=\frac{1}{2}x(1+tanh(u))+\frac{1}{2}x(1-tanh^2(u))c(1+3・0.044715x^2)
$$

### Softmax/CrossEntropy
順伝播
\
| 記号 | 意味 |
| -- | :-- |
| $B$ | バッチ数 |
| $L$ | 誤差関数 |
$$
p_{ib}  = \frac{e^{z_{ib}}}{\sum_je^{z_{jb}}}
\\
L=-\frac{1}{B}\sum_b\sum_it_{ib}\log{p_{ib}}
\\
$$
\
逆伝播
$$
\frac{\partial L}{\partial z}=\frac{p-t}{B}
$$


### Adam
更新手順
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\\
\hat{m}_t =\frac{m_t}{1-\beta_1^t}
\\
\hat{v}_t =\frac{v_t}{1-\beta_2^t}
\\
W_t = W_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

# 変数表記
| 優先度 | prefix | 対象 | 備考 |
| -- | :-- | -- | -- |
| 1 | g_-- | global関数/変数 |  |
| 2 | _-- | メンバー変数 | |
| 3 | c_-- | const | globalで宣言した場合は、prefixはつけず、すべて大文字にすることor defineで宣言すること |
| 4 | lp-- | pointer | |
| 5 | sp-- | shared_ptr | |
| 6 | wp-- | weak_ptr | |
| 7 | n-- | int | |
| 8 | f-- | float | |
| 9 | dbl-- | double | |
| 10 | is/can-- | bool | |
| 11 | str-- | string | |
| 12 | 3文字略省-- | クラス名 | |
| 13 | --複数形 | vector | 本体を複数形にする | |

※変数名本体は意味ごとに大文字から始めること