# cuda_tensor.h 設計仕様

対象: `lib/include/cuda_tensor.h`

## 目的

GPU上の値と勾配をまとめ、自動微分グラフの接続点となるTensorを提供する。

## Tensor

### 内部状態

| メンバー | 形状 | 内容 |
| --- | --- | --- |
| `_mData` | rows × cols | Tensorの値 |
| `_mGrad` | dataと同形状 | 累積勾配 |
| `_spContext` | なし | このTensorを生成した演算Context |

コンストラクタはdataとgradを確保し、gradだけを0へ初期化する。dataは呼び出し側が設定する。

### backward

1. 起点Tensorの全勾配要素を1に設定する。
2. `buildBackwardGraph`で入力側へContextを再帰探索する。
3. 入力側から出力側の順で収集したContext一覧を逆順に走査する。
4. 生存している各出力Tensorから勾配pointerを集める。
5. 各ContextのFunction::backwardを呼ぶ。

通常は1 × 1 loss Tensorを起点にする。非scalar Tensorでも全要素1がseedされるため、結果は「全出力要素の和」の勾配に相当する。任意の外部勾配をseedする公開APIは現在存在しない。

### グラフ探索

各TensorのContextから入力Tensorへ再帰する。visited setはContext pointer単位で重複を除外するため、分岐や複数出力があっても同一Functionのbackwardは1回だけ実行される。

### 勾配蓄積

各Functionのbackwardは通常`+=`相当で入力・Parameter勾配へ加算する。複数経路の勾配を合流できる一方、新しい学習stepの前には`Module::zero_grads`またはOptimizerの`zero_grads`が必要である。

## 所有権

出力TensorはContextをshared pointerで所有する。Contextは入力Tensorをshared pointerで保持し、出力Tensorはweak pointerで参照する。この非対称構造により、逆伝播に必要な入力を保持しつつ出力との循環参照を避ける。

Context内のFunction pointerは非所有である。Functionは計算グラフより長く生存しなければならない。

## 制約

- shape変更APIはなく、構築後のdata/grad形状は固定。
- data、grad、Contextは公開メンバーであり、整合性は利用側の規律に依存する。
- backward後にグラフを明示解放するAPIはない。出力TensorとそのContextが解放されれば、参照されなくなった入力グラフも解放される。
