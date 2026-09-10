# Attention 設計仕様

対象: `include/module_Attention.h`

## 実装状態

公開インターフェース、設定値、学習Parameter、および主要な内部Functionの構成を定義済み。ソース実装は今後追加する。実装には、`Linear`の任意rank対応、動的なhead分割・結合、およびmaskをforwardからbackwardまで安全に保持する仕組みが必要である。

## 目的

Scaled Dot-Product Multi-Head Attentionを提供する。同じクラスをSelf AttentionとCross Attentionの両方に使用し、用途の違いはforwardへ渡すQuery、Key、Valueで表現する。

```text
Self Attention:  Q = K = V = X
Cross Attention: Q = decoder state
                 K = V = encoder output
```

クラス名は簡潔に`Attention`とするが、内部計算は常に`nHeads`個のheadを扱う。`nHeads == 1`も許可する。

## 所有権

Attentionは次の学習Parameterを所有し、`registerParameter()`で登録する。

| 登録名 | shape | 用途 |
| --- | --- | --- |
| `query_weight` | `[E,E]` | Query projection |
| `query_bias` | `[E,1]` | Query bias |
| `key_weight` | `[E,E]` | Key projection |
| `key_bias` | `[E,1]` | Key bias |
| `value_weight` | `[E,E]` | Value projection |
| `value_bias` | `[E,1]` | Value bias |
| `output_weight` | `[E,E]` | head結合後のprojection |
| `output_bias` | `[E,1]` | 出力bias |

`E`はembedding sizeである。Parameterの実体はAttentionが`shared_ptr<Tensor>`で所有し、基底Moduleは非所有pointerのみを登録する。

Query、Key、Value、maskはforward入力であり、Moduleの永続的な状態として保存しない。内部Functionはautogradのbackwardが完了するまで生存する必要があるため、ローカル変数ではなくAttentionのメンバーとして保持する。

## コンストラクタ

```cpp
Attention(
    int nEmbeddingSize,
    int nHeads,
    float fDropoutProbability =0.0f,
    std::uint64_t nDropoutSeed =0
);
```

- `nEmbeddingSize > 0`
- `nHeads > 0`
- `nEmbeddingSize % nHeads == 0`
- `0 <= fDropoutProbability < 1`
- head sizeは`nEmbeddingSize / nHeads`
- dropout seedはAttention内部の乱数系列へ使用する

不正な設定値は`std::invalid_argument`で拒否する。

## 初期化

```cpp
void init(std::mt19937& rngRandom);
```

Weightはfan-inを考慮した分布で初期化し、biasは0で初期化する。forwardより前に1回呼び出す。将来Model共通の初期化規約を導入した場合は、その規約へ統合してよい。

## forward入力

基底Moduleとの互換性のためTensor listを使用する。

| 入力数 | 内容 |
| --- | --- |
| 3 | `{query,key,value}` |
| 4 | `{query,key,value,mask}` |

shape規約は次とする。

```text
query: [E, Lq, B]
key:   [E, Lk, B]
value: [E, Lk, B]
mask:  [Lq, Lk] または [Lq, Lk, H, B]
output:[E, Lq, B]
```

- `Lq`: Query length
- `Lk`: Key/Value length
- `H`: head count
- `B`: batch size

KeyとValueの系列長およびbatch sizeは一致しなければならない。Queryのbatch sizeも一致する必要がある。mask省略時は全位置を参照可能とする。decoder-only GPTでは未来位置を禁止するcausal maskを渡す。

## forward計算

```text
Q = LinearQuery(query)
K = LinearKey(key)
V = LinearValue(value)

Q: [Lq,D,H,B]
K: [D,Lk,H,B]
V: [Lk,D,H,B]

scores  = BatchMatMul(Q,K)
scores  = Scale(scores,1/sqrt(D))
scores  = Mask(scores,mask)       // mask指定時
weights = Softmax(scores,axis=1)
weights = Dropout(weights)        // training時
context = BatchMatMul(weights,V)

context = mergeHeads(context)
output  = LinearOutput(context)
```

`D = E/H`である。head分割・結合はTensorデータをhostへ転送せず、ReshapeとPermute相当のGPU演算で行う。

## backward

Attention全体のbackwardは手書きしない。内部で呼び出したLinear、BatchMatMul、Scale、Mask、Softmax、Dropout、Reshape、Permuteが作成するautograd graphへ委譲する。すべてのParameter勾配は既存値へ加算される。

maskは制御入力であり勾配を計算しない。maskを受け取るFunctionは、そのforwardで使ったmaskをbackwardまで保持しなければならない。別のforwardによるmaskの上書きを参照してはならない。

## training/evaluation

- training時のみattention weightへdropoutを適用する
- evaluation時はdropoutを適用しない
- `Module::setTraining()`で設定された状態をforward時に参照する

## 例外

- 入力数が3または4でなければ`std::runtime_error`
- null入力は`std::invalid_argument`
- embedding、系列長、batch shapeの不一致は`std::invalid_argument`
- mask shape不一致または全key無効行は`std::invalid_argument`
- shape積またはCUDA index上限超過は`std::overflow_error`
- CUDA演算失敗は`std::runtime_error`

## 完了条件

- Self AttentionとCross Attentionを同じクラスで処理できる
- causal maskによって未来位置のweightが0になる
- 各headを独立に計算し、結合後shapeが`[E,Lq,B]`になる
- Q/K/V/Oの全Parameterへ勾配が流れる
- input、Parameterの勾配が数値微分と一致する
- training/evaluationでdropoutの有無が切り替わる
- Parameterがstate dictへ登録され、保存・読込できる
