# cuda_bublas.h 設計仕様

対象: `lib/include/cuda_bublas.h`

## 目的

cuBLAS handleの生成と破棄をRAIIで管理し、プロセス内で共有するhandleを提供する。

## cuCublas

### ライフサイクル

コンストラクタで`cublasCreate`、デストラクタで`cublasDestroy`を呼ぶ。`get()`は保持中の`cublasHandle_t`を返す。

handleはクラス外へ貸し出すだけで、呼び出し側は破棄してはならない。

### 現在の制約

- create/destroyの戻り値を検査していない。
- コピー・ムーブ操作を明示的に禁止していないため、インスタンスをコピーして使用してはならない。
- streamを明示設定していないため、既定stream上の演算を前提とする。

## getCublasHandle

関数ローカルstaticの`cuCublas`を使い、初回呼び出し時にhandleを生成して以後共有する。C++11以降のstatic初期化規則により、初期化自体はthread-safeである。

この関数はヘッダー内定義だが`inline`指定がない。現在は単一翻訳単位から利用しているが、複数翻訳単位で直接includeするとODR違反になる可能性があるため、将来は`inline`化または実装ファイルへの移動が必要である。

## 利用箇所

`cuda_matrix.cpp`のcuBLASベース演算がhandleを取得する。代表例はscale、AXPY、GEAM、GEMM、transposeである。
