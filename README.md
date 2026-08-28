# ai-cpp-from-scratch

CUDA と C++ を使い、ニューラルネットワークの基本要素を低レベルから実装する開発中のプロジェクトです。行列演算、Tensor の自動微分、活性化関数、損失関数、Optimizer など、学習基盤を構成する部品を実装しています。

Graph と Model は派生クラスでネットワーク構造やパラメータを定義することを前提とした基盤です。エンドツーエンドで学習を実行するプログラムはまだ実装されておらず、このリポジトリは完成済みの学習アプリケーションではありません。

## 実装済みの主な機能

- CPU の `Mat` と CUDA デバイス上の `cuMat` による行列データ管理
- cuBLAS および CUDA kernel を使った行列演算
- `Tensor`、`Function`、`Context` による逆伝播の基盤
- Linear、ReLU、GELU、Softmax Cross Entropy
- SGD、Adam
- Graph/Model の保存、読み込み、勾配初期化のための基底 API
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
│   ├── include/   行列・Tensor・Graph・Optimizerの公開ヘッダー
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
