# 設計仕様書

このディレクトリには、公開ヘッダーごとの設計仕様をまとめる。仕様は現在の宣言と実装を基準とし、責務、所有権、データ形状、API契約、例外条件、利用上の注意を記載する。

## アプリケーション層

| ヘッダー | 設計仕様 |
| --- | --- |
| `include/mnist.h` | [mnist.md](mnist.md) |
| `include/neuralnet.h` | [neuralnet.md](neuralnet.md) |
| `include/trainer.h` | [trainer.md](trainer.md) |

## ライブラリ層

| ヘッダー | 設計仕様 |
| --- | --- |
| `lib/include/cuda_bublas.h` | [cuda_bublas.md](cuda_bublas.md) |
| `lib/include/cuda_function.h` | [cuda_function.md](cuda_function.md) |
| `lib/include/cuda_matrix.h` | [cuda_matrix.md](cuda_matrix.md) |
| `lib/include/cuda_tensor.h` | [cuda_tensor.md](cuda_tensor.md) |
| `lib/include/matrix.h` | [matrix.md](matrix.md) |
| `lib/include/module.h` | [module.md](module.md) |
| `lib/include/optimizer.h` | [optimizer.md](optimizer.md) |

## 共通規約

- ニューラルネットワークの行列は「特徴量またはクラス数 × バッチ数」で表す。
- `Mat`と`cuMat`はcolumn-majorであり、要素位置は`column * rows + row`である。
- `Tensor`のデータと勾配はGPU上に置く。
- 学習ParameterとBufferの所有権は派生`Module`が持ち、基底`Module`は非所有pointerを登録する。
- CUDA演算のbackwardは、原則として既存の勾配へ加算する。

### 命名規則

変数名は型や所有関係を示すprefixと、意味を示す大文字始まりの本体を組み合わせる。constやpointerなど複数の性質がある場合は、`c_spmInput`のようにprefixを重ねる。

| 優先度 | prefix・形式 | 対象 |
| --- | --- | --- |
| 1 | `g_` | global関数・変数 |
| 2 | `_` | メンバー変数 |
| 3 | `c_` | const。global定数はprefixを付けず全大文字、またはmacroとする |
| 4 | `lp` | raw pointer |
| 5 | `sp` | `shared_ptr` |
| 6 | `wp` | `weak_ptr` |
| 7 | `n` | 整数 |
| 8 | `f` | `float` |
| 9 | `dbl` | `double` |
| 10 | `is` / `can` | `bool` |
| 11 | `str` | 文字列 |
| 12 | クラス名の3文字略称 | クラス型の変数 |
| 13 | 本体を複数形 | `vector`などのコレクション |

### コーディングスタイル

- インデントは半角スペース4個とし、タブ文字は使用しない。
- `namespace`、`class`、`struct`、関数、制御文の開始波括弧は宣言または条件の次行に置く。
- `if`、`else`、`for`、`while`などの制御文は、本体が1文だけでも必ず波括弧で囲む。
- 1行には原則として1文だけを記述する。
- pointerとreferenceの記号は型側へ寄せ、`Tensor* lpTensor`、`const cuMat& c_mValue`のように記述する。
- 代入演算子と二項演算子の前後、およびカンマの後には半角スペースを置く。
- 空でない丸括弧の内側には半角スペースを置く。空の引数リストは`()`と記述する。
- 長い関数宣言、関数呼出し、条件式は意味のまとまりで改行し、継続行もスペースで整列する。
- formatterを使用する場合も、インデント幅4、タブ不使用、Allman形式、制御文への波括弧追加を維持する。

```cpp
void updateTensor( Tensor* lpTensor, const cuMat& c_mGradient )
{
    if( lpTensor == nullptr )
    {
        throw std::invalid_argument( "tensor must not be null" );
    }

    for( int nRow = 0; nRow < lpTensor->_mData._nRows; ++nRow )
    {
        updateRow( lpTensor, c_mGradient, nRow );
    }
}
```
