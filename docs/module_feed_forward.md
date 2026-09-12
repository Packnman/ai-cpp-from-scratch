# FeedForward 設計仕様

対象ヘッダ: `model/include/module_FeedForward.h`。実装: `model/src/module_FeedForward.cpp`。

## 責務・公開 API・設定

実装済みの位置ごとに独立した2層 FFN。`FeedForward( int nEmbeddingSize, int nHiddenSize, float fDropoutProbability = 0.0f, std::uint64_t nDropoutSeed = 0 )`、`init( std::mt19937& rngRandom )`、`forward( TensorList& spmInputs )` を提供する。E、F は正数、dropout は有限かつ `[0,1)`。構築後、forward の前に `init` が必要。重みは標準偏差 `1/sqrt(fan_in)` の正規分布、bias は0。

## Tensor と計算

入力は1個、rank 2以上の連続 `[E,D1,...,Dn]`。Linear が後続軸をまとめて GEMM し、形状を復元する。中間 `[F,D1,...,Dn]`、出力 `[E,D1,...,Dn]`。

`U=W1 X+b1` → `G=GELU(U)` → `Y=W2 G+b2` → 学習時のみ inverted dropout。既存 GELU の tanh 近似を使用する。LayerNorm・残差はブロックの責務。

逆順に dropout、Linear2、GELU、Linear1 の backward を実行する。`dW2 += dY G^T`、`dG = W2^T dY`、`dU = dG * GELU'(U)`、`dW1 += dU X^T`、`dX += W1^T dU`。bias は位置軸で総和する。

## 所有権・保存・モード

`weight1:[F,E]`、`bias1:[F,1]`、`weight2:[E,F]`、`bias2:[E,1]` を FFN が shared_ptr で所有し Module に非所有登録する。入力は Context が保持し、モデルと Parameter は backward まで生存・不変が必要。Linear、GELU はメンバー。dropout は各グラフの Context が所有し、別 forward で上書きしない。評価モードでは dropout を省略。乱数状態は保存せず、Parameter は既存 Model v2、構造は会話 bundle v1 で保存する。

## 例外と完了条件

不正な設定、入力数、null、rank、特徴数、空の位置軸、非連続入力は `invalid_argument`。CUDA 失敗は基盤例外を伝播する。

`conversation_check` で別形状 forward を挟んだ dropout backward と同 seed の参照を比較する。Transformer の過学習と保存後 logits 一致で2層双方への学習と永続化を確認する。分類モデルの既存テストで rank 2 Linear 互換性を確認する。
