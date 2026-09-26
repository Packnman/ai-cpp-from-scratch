# ai_cpp

CUDAを利用したC++20の機械学習基盤です。CPU行列、CUDAメモリ・行列・テンソル、
ニューラルネットワーク向け演算、モジュール、最適化器、文字トークナイザーを
静的ライブラリ `ai_cpp` として提供します。

現在のスナップショットはライブラリ実装の整理段階です。`app/main.cpp` は以前の
Agent実装を参照するため、現行のCMakeビルドには含めていません。また、現時点では
テストプログラムもリポジトリに含まれていません。

## 主な機能

- CPU行列とEuler角・Quaternionの演算
- CUDAメモリ、行列、テンソルの管理
- Linear、Embedding、BatchNorm、LayerNorm、Conv2D、Pooling
- ReLU、GELU、Dropout、Softmax、各種損失関数
- Add、BatchMatMul、Mask、Permute、Reshape、Scale
- SGD、Adam最適化器
- 文字単位トークナイザー

公開ヘッダーごとの資料は [`lib/docs/`](lib/docs/) にあります。

## 必要環境

- CMake 3.18以上
- C++20対応コンパイラ
- CUDA Toolkit 11.2以上
- Ninja（以下の例で使用。別のCMake generatorも利用可能）

既定のCUDA architectureはAmpere世代の `86` です。使用するGPUに合わせて
`AI_CPP_CUDA_ARCHITECTURES`を変更してください。

## ビルド

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DAI_CPP_CUDA_ARCHITECTURES=86
cmake --build build
```

生成される主な成果物は `build/libai_cpp.a` です。複数architectureを指定する場合は、
値をセミコロン区切りにします。

```sh
cmake -S . -B build -G Ninja \
  -DAI_CPP_CUDA_ARCHITECTURES="86;89"
```

## ディレクトリ構成

```text
lib/include/  公開ヘッダー
lib/src/      C++ / CUDA実装
lib/docs/     ライブラリ資料
app/          現行ビルド対象外の旧Agentエントリーポイント
```

`data/`、`models/`、`experiments/` はローカルのデータセット、学習済みモデル、
実験結果の保存先としてGit管理対象外にしています。
