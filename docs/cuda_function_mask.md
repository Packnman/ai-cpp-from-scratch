# Mask 設計仕様

対象: `lib/include/cuda_function_Mask.h`

## 実装状態

コンストラクタとFunctionメソッドのソース基盤のみ存在する。現在はmaskを保持せず、`forward` と `backward` が `std::logic_error` を送出する。

## 目的

attention scoreのうち参照を許可しない位置を無効化する。causal maskやpadding maskをSoftmaxの直前へ適用する用途を想定する。

## mask規約

v1では浮動小数点Tensorをbinary maskとして扱う。

```text
mask == 0: 参照禁止
mask != 0: 参照可能
```

maskは制御情報であり微分しない。現行の `Mask(Tensor* lpMask)` を維持する場合、MaskはTensorへの非所有pointerを保持し、そのTensorの `_mGrad` は更新しない。将来は整数またはbool相当の専用Tensorへ変更してもよい。

attention scoreを次のshapeとする。

```text
scores: [queryLength, keyLength, heads, batch]
```

maskは次のいずれかを受け付ける。

| mask shape | 意味 |
| --- | --- |
| scoresと完全に同じshape | 要素ごとのmask |
| `[queryLength, keyLength]` | headsとbatchへbroadcastする共通mask |

v1では上記以外の暗黙broadcastを行わない。

## Forward

入力scoreを $S$、maskを $M$ とすると次を計算する。

$$
Y_i=
\begin{cases}
S_i & M_i\ne0\\
-\infty & M_i=0
\end{cases}
$$

出力は入力と同shapeの新しいTensorとする。各query・head・batchには最低1個の参照可能なkeyが必要である。全keyがmaskされた行は後段Softmaxで未定義になるため、検証して拒否する。

## Backward

上流勾配を $G$ とすると入力勾配は次のとおりである。

$$
\frac{\partial L}{\partial S_i}=
\begin{cases}
G_i & M_i\ne0\\
0 & M_i=0
\end{cases}
$$

入力の既存勾配へ加算する。maskの勾配は計算しない。

## 所有権と更新

- Maskはmask Tensorを所有しない
- mask Tensorはforwardとbackwardが完了するまで生存する
- 同じMaskインスタンスでmask内容を更新する場合、forwardからbackwardまで内容を変更してはならない
- 安全性が必要ならContextへmaskまたはmaskのsnapshotを保存する設計へ拡張する

## 検証と例外

- 入力数が1でなければ `std::runtime_error`
- mask pointerがnullなら `std::invalid_argument`
- scoreとmaskが対応しないshapeなら `std::invalid_argument`
- 全keyが無効な行があれば `std::invalid_argument`
- backwardの勾配shapeがscoreと異なれば `std::invalid_argument`

## 完了条件

- causal maskで未来位置のscoreが無効になる
- 2次元maskがheadsとbatchへ正しくbroadcastされる
- masked位置から入力勾配が流れない
- mask Tensorの勾配を変更しない
- 全keyがmaskされた行と不正shapeを拒否する
