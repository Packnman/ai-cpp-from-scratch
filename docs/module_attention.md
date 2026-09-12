# Attention 設計仕様

対象ヘッダ: `model/include/module_Attention.h`。実装: `model/src/module_Attention.cpp`。

## 責務と公開 API

実装済みの因果 Multi-Head Self-Attention。`Attention( int nEmbeddingSize, int nHeads, float fDropoutProbability = 0.0f, std::uint64_t nDropoutSeed = 0 )`、`init( std::mt19937& rngRandom )`、`forward( TensorList& spmInputs )`を提供する。入力は1個の `[E,T,B]`、出力も `[E,T,B]`。以前の未実装設計にあった3入力の Cross Attention は今回の契約に含めない。

`E > 0`、`H > 0`、`E % H == 0`、有限な `0 <= dropout < 1` が必要。系列長とバッチは正数。構築後に `init` を呼ぶ。Transformer は内部で初期化する。重みは平均0、標準偏差 `1/sqrt(E)` の正規分布、bias は0。

## 形状と処理順序

1. Linear で Q、K、V を `[E,T,B]` に射影。
2. `[H,D,T,B]` に reshape (`D=E/H`) し、Q=`[T,D,H,B]`、K=`[D,T,H,B]`、V=`[T,D,H,B]` に permute。
3. `S[q,k,h,b] = Σd Q[q,d,h,b] K[d,k,h,b] / sqrt(D)`。
4. `k > q` を負の無限大で mask。key 軸（axis=1）で Softmax。
5. 学習時は注意確率に inverted dropout (`mask/(1-p)`)。
6. `C[q,d,h,b] = Σk A[q,k,h,b] V[k,d,h,b]`。
7. `[H,D,T,B]` へ戻して `[E,T,B]` に結合し、出力 Linear。

データセットは右 PAD のみを作るため、有効 query が PAD key を参照することはない。任意の左 PAD には対応しない。推論では PAD を入力しない。

## 所有権・寿命と勾配

`query_weight`、`key_weight`、`value_weight`、`output_weight` は `[E,E]`、対応する `*_bias` は `[E,1]`。Attention が shared_ptr で所有し、Module は非所有登録する。重み・bias の変更や破棄は、その forward の backward 完了後に行う。

固定設定の演算はメンバーとして保持する。可変 reshape、permute、因果 mask、学習 dropout は forward ごとに作り、`Context::_spFunction` が所有する。mask を扱う演算は mask Tensor 自体も所有する。次の forward による上書きはない。グラフ破棄で解放し、循環所有は作らない。

逆伝播は既存 Function を合成する。`dQ = dS K^T / sqrt(D)`、`dK = Q^T dS / sqrt(D)`、`dV = A^T dC`、Softmax は `dS = A * (dA - Σkey A*dA)`。mask の禁止位置の勾配は0。入力・全 Parameter に加算する。mask と整数 shape に勾配はない。

## 学習・評価、保存、例外

`setTraining(false)` で dropout を省略。学習 forward ごとに seed を進める。乱数状態は保存しない。Parameter は Model v2 の名前・形状付き FP32 保存に含まれる。会話 bundle v1 には構造設定も記録する。

設定、入力数、null、rank、特徴数の不正は `invalid_argument`。基盤の演算で形状積上限や CUDA 失敗も拒否する。`init` と Model の寿命を呼出し側が管理する。

## テストによる完了条件

`conversation_check` で2ヘッド・2バッチの入力と全 Parameter の中心差分勾配（許容誤差0.002）、未来入力変更に対する過去出力の不変性、短い別 forward を挟んだ backward を確認する。`transformer_functions_check` で基盤演算の勾配、mask、Softmax を確認する。保存は Transformer 全体の logits 一致で確認する。
