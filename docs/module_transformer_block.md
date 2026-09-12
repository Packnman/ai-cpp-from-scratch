# TransformerBlock 設計仕様

対象ヘッダ: `model/include/module_TransformerBlock.h`。

## 責務・API・設定・形状

`TransformerBlock( int nEmbedding, int nHeads, int nHidden, float fDropout = 0.1f, std::uint64_t nSeed = 42 )`、`init( std::mt19937& )`、`forward( TensorList& )` を提供する。入力は1個の `[E,T,B]`、出力は同形状。E、H、F の制約は Attention／FFN と同じ。init は gamma を1、beta を0、Attention／FFN の重みを各 fan-in に応じて初期化する。

## 処理順序・数式・勾配

`Y = X + Attention(LN1(X))`、`Z = Y + FFN(LN2(Y))` の Pre-LN 構成。LayerNorm は各 `[t,b]` の特徴軸で `mu=mean(X)`、`var=mean((X-mu)^2)`、`LN(X)=gamma*(X-mu)/sqrt(var+1e-5)+beta` を計算する。

残差の backward は恒等経路と変換経路の勾配を加算する。LayerNorm の入力・gamma・beta、Attention／FFN の全 Parameter に勾配が流れる。整数や mask には勾配はない。

## 所有権・寿命・モード・保存

4個の `[E,1]` Parameter `gamma1/beta1/gamma2/beta2` と2子 Module を値として所有し、Module に非所有登録する。演算は入力グラフを Context に保持する。ブロック本体・Parameter は backward 完了まで生存し、更新しない。複数 forward の可変状態は子 Module のグラフごとに保持する。`setTraining` は子に伝播し、評価時は2か所の dropout が無効となる。

Model v2 の名前付き重みとして保存する。会話 bundle v1 はブロック数・構造も保存する。Optimzer 状態や RNG は含めない。

## 例外・完了条件

不正入力／設定は LayerNorm、Attention、FFN の `invalid_argument` 等を伝播する。CUDA 失敗も伝播する。`conversation_check` の全体過学習と保存後 logits 一致、Attention 勾配・因果性、dropout 寿命検証を完了条件とする。
