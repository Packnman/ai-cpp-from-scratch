# matrix.h 設計仕様

対象: `lib/include/matrix.h`

## 目的

CPU上のcolumn-major float行列と、ベクトル・Euler角・Quaternionに関する補助演算を提供する。

## 共通レイアウト

`Mat(row, col)`は`col * rows + row`でアクセスする。要素範囲の検査は行わないため、呼び出し側が有効な行・列を保証する。

## Mat

### 所有権と値型

`_lpfHost`が`malloc`されたCPUメモリを所有し、デストラクタで`free`する。コピーはdeep copy、ムーブはpointer移譲である。構築直後の要素は未初期化。

### API

| API | 仕様 |
| --- | --- |
| `ones` | 全要素を1へ設定 |
| `tri` | 正方行列のtraceを返す |
| `trp` | 転置行列を新規生成 |
| `toEul` | 3 × 3 DCMをEuler角へ変換 |
| `toQtn` | 3 × 3 DCMをQuaternionへ変換 |
| `operator(row,col)` | 要素参照 |
| `+=`, `-=` | 同形状行列との要素単位演算 |
| `*=(Mat)` | 行列積で自身を置換 |
| `*=(float)` | 全要素をscale |

非メンバー`+`、`-`、`*`は新しい行列を返す。形状不一致は例外となる。

## Vec

`Mat(nRows, 1)`を表す派生型。

- `skew`: 3要素ベクトルの交代行列
- `cpx`: 3次元cross product
- `dot`: 同じ長さの内積
- `norm`: Euclidean norm
- `exp(dt)`: 3次元角速度ベクトルからQuaternion増分を生成
- `operator(row)`: 1列目への簡略アクセス

`exp`はnormで除算するため、ゼロベクトル入力は現在の実装では未定義の数値結果になり得る。

## Euler

3要素Vec。要素順序は実装上、index 0がroll、1がpitch、2がyawとしてDCM変換に使われる。

- `toDCM`: Euler角を3 × 3方向余弦行列へ変換
- `toQtn`: DCMを経由してQuaternionへ変換
- Euler同士の`operator*`: DCM積をEulerへ戻す
- EulerとVecの`operator*`: DCMによるベクトル回転

## Quaternion

4要素Vec。配列順序はvector part `[x, y, z]`、scalar part `w`である。

| API | 仕様 |
| --- | --- |
| `cnj` | 共役Quaternion |
| `axis` | vector partを返す |
| `dot_E` | Quaternion微分用行列 |
| `dot(rps)` | 角速度からQuaternion微分を計算 |
| `normalized` | 自身を単位長へ正規化 |
| `toDCM` | 3 × 3方向余弦行列へ変換 |
| `toEul` | DCMを経由してEulerへ変換 |

Quaternion積とQuaternionによるVec回転をfriend演算子として提供する。

## 例外と注意

サイズ依存APIは要求形状を満たさない場合に`std::runtime_error`を送出する。インデックス範囲、malloc失敗、一部の三角関数入力やゼロnormは明示検査していない。
