# Transformer 設計仕様

対象ヘッダ: `model/include/model_transformer.h`。実装: `model/src/model_transformer.cpp`。

## 責務と API

`TransformerConfig` と整数 ID を入力する decoder-only `Transformer : Model` を提供する。

| 設定 | 既定値 | 制約 |
| --- | --- | --- |
| nVocabulary | 0（必須設定） | 7以上 |
| nBlocks | 4 | 正数 |
| nEmbedding | 256 | 正数、ヘッド数で割り切れる |
| nHeads | 4 | 正数 |
| nHidden | 1024 | 正数 |
| nContext | 128 | 正数 |
| fDropout | 0.1 | 有限、0以上1未満 |
| nSeed | 42 | uint64_t、初期化時 mt19937 へ変換 |

`validate()` は設定違反を `invalid_argument` で拒否する。`Transformer( const TransformerConfig& )` は初期化まで実行する。`config()` は読み取り専用参照を返す。

`forward( const std::shared_ptr<const cunMat>& )` は int32 `[T,B]` から FP32 logits `[V,T,B]` を返す。`1 <= T <= nContext`、`B > 0`、全 ID は `[0,V)`。`loss( ids, targets, nPadId = 0 )` は教師 `[T,B]` を使い scalar `[1,1]` を返す。浮動小数点 `TensorList` の override は誤用を `invalid_argument` で拒否する。従来の分類 API は変更しない。

## 処理・数式・勾配

`X[e,t,b] = token_weight[e,ids[t,b]] + position_weight[e,t]` → Pre-LN ブロックを4回 → 最終 LayerNorm →語彙 Linear。token embedding `[E,V]` と出力重み `[V,E]` は別 Parameter。位置重みは `[E,C]`。位置は各入力の先頭から0にリセットする。学習と推論で同じ契約。

各 embedding の勾配は参照された ID ごとに加算、位置勾配はバッチ方向にも加算する。最終 LayerNorm は epsilon `1e-5`、gamma=1、beta=0。Linear の weight は標準偏差 `1/sqrt(E)` の正規分布、bias=0。損失は [整数教師用仕様](cuda_function_index_cross_entropy.md) に従う。

## 所有権・寿命・モード

Transformer は埋め込み、正規化、出力 Parameter を値として、ブロックを unique_ptr で所有する。Module の登録は非所有。forward は入力 ID を deep copy し、位置 ID とともに各 IndexFunction Context が保持する。呼出し側は元 ID を変更できる。計算グラフの backward が完了するまでモデルと重みを保持し、更新しない。可変長の複数 forward を保持できる。学習・評価モードは子 Module に伝播し、dropout のみが変わる。

## 保存・互換性と例外

継承した `save/load` は Model v2 の重みのみを扱い、同じ設定のモデルが必要。設定・語彙を含む推論には [会話 runtime](conversation_runtime.md) の bundle API を使う。完全な Optimizer 再開は対象外。

不正 shape、null、ID 範囲外、教師 shape 不一致を拒否し、CUDA の失敗を伝播する。メモリ要求は設定に応じて増加し、デバイス確保に失敗すれば例外となる。

## 完了条件

`conversation_check` の短系列・小規模過学習（80更新で loss < 0.1）、保存再読込の FP32 logits 完全一致、学習／評価／最良重みの test 評価が通ること。既定4ブロック・256・4ヘッド・1024・T128・B8の実データ検証は [実測記録](conversation_validation.md) に記録する。
