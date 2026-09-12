# 整数教師用交差エントロピー設計仕様

対象ヘッダ: `lib/include/cuda_function_IndexCrossEntropy.h`。実装: 同名 `.cu`。

## 責務・公開 API・形状

`IndexCrossEntropy( const cunMat& c_mTargets, int nPadId = 0 )`、`forward( const TensorList& )`、`backward( gradients, inputs, outputs )` を提供する。浮動小数点の既存 `SoftmaxCrossEntropy` は変更しない。

logits は FP32 連続 `[V,D1,...,Dn]`、教師は int32 `[D1,...,Dn]`、出力は scalar `[1,1]`。Transformer では `[V,T,B]` と `[T,B]`。クラス ID は `[0,V)`、除外 ID は設定した PAD のみ。

## 数式・CUDA 処理

位置 i の `m_i=max_v z_vi`、`s_i=Σv exp(z_vi-m_i)`、`p_vi=exp(z_vi-m_i)/s_i` を計算する。1位置1 block、128 threads の reduction を使用する。

`N = count(y_i != PAD)`、`L = Σvalid (log(s_i)+m_i-z[y_i,i])/N`。出力への集約は FP32 atomicAdd なので加算順による最下位桁の差を許す。

`dz_vi += upstream * (p_vi - [v=y_i])/N`。PAD 位置は勾配0。全 PAD は loss=0、勾配0と定義する。教師に勾配はない。

## 所有権・寿命・モード・例外

構築時に教師を deep copy する。forward で確率を保持するため、1 instance につき forward 1回。Transformer の loss は毎回新規 instance を作り、Context が所有する。元教師の変更は過去の backward に影響しない。入力 logits は Context が保持し、モデルは backward まで生存・不変が必要。

学習／評価の計算式に違いはない。評価では backward を呼ばない。損失状態は保存形式に含めず、重み保存との互換性に影響しない。

再利用、不正入力数、null、空 Tensor、非連続、rank/shape 不一致、INT_MAX 超過は `invalid_argument`。教師範囲外は `out_of_range`、CUDA launch 失敗は `runtime_error`。入力 logits が有限であることは呼出し側の前提で、学習 API は非有限損失を拒否する。

## 完了条件

`conversation_check` で CPU の log-sum-exp 参照との一致、全クラスの解析勾配、PAD 除外、全 PAD、元教師の変更、不正 ID の拒否を確認する。
