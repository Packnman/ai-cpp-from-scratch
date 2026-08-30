# cuda_matrix.h 設計仕様

対象: `lib/include/cuda_matrix.h`

## 目的

GPUメモリ上の2次元float行列を所有する`cuMat`と、cuBLASまたはCUDA kernelで動作する低レベル演算APIを提供する。

## メモリレイアウト

行列はcolumn-majorで格納する。

```text
index(row, col) = col * rows + row
```

`IDX2F`はcolumn-major、`IDX2C`はrow-majorのインデックス計算macroである。ニューラルネットワークでは列をバッチ、行を特徴量またはクラスとして使う。

## cuMat

### 所有権

`_lpfDevice`がCUDA device memoryを所有し、デストラクタで`cudaFree`する。`_nRows`と`_nCols`が論理形状を表す。

### 構築・コピー・ムーブ

- デフォルト構築: 0 × 0、device pointerはnull
- サイズ指定構築: `rows * cols * sizeof(float)`を確保
- コピー: 同じ形状を確保してDevice-to-Deviceコピー
- ムーブ: pointerと形状を移譲し、移動元を空にする
- コピー代入: 必要なら再確保してDevice-to-Deviceコピー
- ムーブ代入: 現在の領域を解放して所有権を移譲

確保直後の値は未初期化である。必要に応じて`cuda_fill`を呼ぶ。

### CPU転送

現在の命名は転送方向と直感が逆なので注意する。

| API | 実際の方向 |
| --- | --- |
| `upload(Mat& host) const` | Device → Host |
| `download(const Mat& host)` | Host → Device |

呼び出し側はHostとDeviceの形状を一致させる必要がある。現実装の転送関数はCUDAエラーを検査しない。

## 基本演算

| API | 意味 |
| --- | --- |
| `cuda_axpy` | result += alpha * A |
| `cuda_geam` | result = alpha * A + beta * B |
| `cuda_gemm` | result = alpha * op(A) * op(B) + beta * result |
| `cuda_fill` | 全要素を指定値に設定 |
| `cuda_transpose` | result = transpose(A) |
| `cuda_scale` | result *= scalar |
| `cuda_mul_elementwise` | result = A element-wise-multiply B |

行列サイズの一致や積の内側次元は各演算の前提条件であり、実装が検査するAPIでは不一致時に`std::runtime_error`を送出する。

## ニューラルネットワークkernel API

### ReLU / GELU

forwardは入力と同形状の出力を生成する。backwardは上流勾配へ導関数を掛け、入力勾配へ加算する。

### Dropout

`cuda_Dropout_forward`はcuRANDが生成した一様乱数を保持するmaskを、0または`1 / (1-p)`へ変換しながら入力へ掛ける。backwardは保存済みmaskを上流勾配へ掛けて入力勾配へ加算する。

### BatchNorm

入力形状はfeatures × batch、gamma、beta、running統計、逆標準偏差はfeatures × 1である。

- training forward: バッチ平均・分散を計算し、running統計を更新して正規化
- evaluation forward: running統計で正規化し、running統計は更新しない
- backward: input、gamma、betaの勾配を加算

学習時の正規化は分母Bの分散を使用し、running variance更新にはBが2以上なら不偏分散補正を使用する。

### SoftmaxCrossEntropy

logitsとone-hot targetはclasses × batch。forwardはバッチ平均lossを1 × 1へ出力し、数値安定化のため各列の最大logitを差し引く。backwardはlogits勾配へ`upstream * (softmax - target) / batch`を加算する。

### Adam

パラメータ、勾配、1次・2次モーメントは同形状。bias correction済みの係数を使い、パラメータをin-place更新する。

## 実行モデルとエラー

kernel launchは非同期で、各wrapperは主に`cudaGetLastError`でlaunch errorを検査する。完了時エラーをその場で必ず検出する同期APIではない。演算順序は同一の既定CUDA streamを前提とする。
