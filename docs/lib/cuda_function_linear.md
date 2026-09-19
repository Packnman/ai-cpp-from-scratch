# Linear 設計仕様

対象ヘッダ: `lib/include/cuda_function_Linear.h`。公開 API は既存どおり `Linear( Tensor* lpWeight, Tensor* lpBias )`、Function の forward/backward。

## 責務・設定・形状

任意 rank（2以上）の特徴軸へ線形変換を適用する。入力1個 `[I,D1,...,Dn]`、weight `[O,I]`、bias `[O,1]`、出力 `[O,D1,...,Dn]`。位置数 `P=Π Dk` をまとめ、既存2次元 GEMM を使う。rank 2 の分類 API と結果を維持する。学習／評価で違いはなく、既定の学習設定も持たない。

## 処理と勾配

入力 `[I,P]` の reshape view → `Y=W X` → `Y+=bias*ones[1,P]` →出力形状を復元。backward は入力と出力の形状からその都度2次元 view を作り、`dX+=W^T dY`、`dW+=dY X^T`、`db+=dY ones^T` を実行する。

## 所有権・寿命・例外

weight と bias は非所有 pointer。モデル側が backward まで所有し、値を変更しない。Context が入力を保持する。ones はその呼出しのローカル Tensor。公開 `_mTmp` は互換性のため残すが新実装は使用しない。別 shape の forward を挟んでも backward に共有可変 shape は不要。

null、入力数、rank、特徴数、bias shape、空位置、非連続、INT_MAX を超える入力は `invalid_argument`。CUDA 失敗は基盤例外を伝播する。演算自身に保存状態はなく、Parameter は所有 Model が保存する。

## 完了条件

既存分類テスト全件、`conversation_check` の rank 3 Attention 全 Parameter 数値勾配・別 shape forward、Transformer 過学習を完了条件とする。
