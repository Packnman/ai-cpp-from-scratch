# FeedForward 設計仕様

対象: `include/module_FeedForward.h`

## 実装状態

公開インターフェース、設定値、学習Parameter、および内部Functionの構成を定義済み。ソース実装は今後追加する。実装前に`Linear`を先頭特徴量軸以外の任意rankへ対応させる。

## 目的

Transformer block内のposition-wise Feed-Forward Networkを提供する。系列位置とbatchを互いに混ぜず、各位置へ同じ2層の全結合変換を適用する。

```text
Y = Linear2(GELU(Linear1(X)))
```

training時はLinear2の出力へdropoutを適用する。残差加算とLayerNormはTransformer block側の責務とし、FeedForwardには含めない。

## 所有権

FeedForwardは次の学習Parameterを所有し、`registerParameter()`で登録する。

| 登録名 | shape | 用途 |
| --- | --- | --- |
| `weight1` | `[F,E]` | embeddingからhiddenへの変換 |
| `bias1` | `[F,1]` | 第1層bias |
| `weight2` | `[E,F]` | hiddenからembeddingへの変換 |
| `bias2` | `[E,1]` | 第2層bias |

- `E`: embedding size
- `F`: hidden size。一般的には`4E`前後

Parameterの実体はFeedForwardが`shared_ptr<Tensor>`で所有し、基底Moduleは非所有pointerのみを登録する。forward入力はメンバーへ保存しない。

Linear、GELU、Dropoutはautogradのbackwardまで生存する必要があるため、FeedForwardのメンバーとして保持する。

## コンストラクタ

```cpp
FeedForward(
    int nEmbeddingSize,
    int nHiddenSize,
    float fDropoutProbability =0.0f,
    std::uint64_t nDropoutSeed =0
);
```

- `nEmbeddingSize > 0`
- `nHiddenSize > 0`
- `0 <= fDropoutProbability < 1`

不正な値は`std::invalid_argument`で拒否する。

## 初期化

```cpp
void init(std::mt19937& rngRandom);
```

Weightは各層のfan-inを考慮した分布で初期化し、biasは0で初期化する。forwardより前に1回呼び出す。

## shape規約

```text
input:   [E, D1, ..., Dn]
hidden:  [F, D1, ..., Dn]
output:  [E, D1, ..., Dn]
```

Transformerでの代表shapeは次である。

```text
input/output: [embeddingSize, sequenceLength, batch]
hidden:       [hiddenSize, sequenceLength, batch]
```

先頭軸だけをLinear変換し、後続軸は位置として保持する。系列位置間の情報交換はAttentionの責務であり、FeedForwardでは行わない。

## forward

入力Tensorは正確に1個とする。

```text
hidden = Linear1(input)
hidden = GELU(hidden)
output = Linear2(hidden)
output = Dropout(output)  // training時のみ
```

GELUはGPT系モデルの標準的な活性化として使用する。ReLUへ変更する場合はコンストラクタ設定または別Moduleとして明示する。

## backward

FeedForward全体のbackwardは手書きしない。Linear、GELU、Dropoutが作成するautograd graphへ委譲する。

```text
output gradient
→ Dropout backward
→ Linear2 backward
→ GELU backward
→ Linear1 backward
→ input gradient
```

input、weight、biasのすべての勾配は既存値へ加算する。

## training/evaluation

- training時のみ第2Linearの出力へdropoutを適用する
- evaluation時はdropoutを適用しない
- `Module::setTraining()`で設定された状態をforward時に参照する

## 検証と例外

- 入力数が1でなければ`std::runtime_error`
- 入力がnullなら`std::invalid_argument`
- 入力rankが1未満、または先頭軸が`E`と異なれば`std::invalid_argument`
- shape積またはCUDA index上限超過は`std::overflow_error`
- CUDA演算失敗は`std::runtime_error`

## 完了条件

- rank 2、3、4で後続shapeを保持する
- 各系列位置へ独立に同じ変換を適用する
- forwardがCPU参照計算と一致する
- input、weight1、bias1、weight2、bias2の勾配が数値微分と一致する
- training/evaluationでdropoutの有無が切り替わる
- Parameterがstate dictへ登録され、保存・読込できる
