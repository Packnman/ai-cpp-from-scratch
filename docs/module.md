# module.h 設計仕様

対象: `lib/include/module.h`

## 目的

学習パラメータ、永続Buffer、子Moduleを名前付きで登録し、モデル全体の状態列挙、モード伝播、勾配初期化、保存・読込を提供する。

## NamedTensor / StateDict

`NamedTensor`は状態名と非所有`Tensor*`の組である。`StateDict`はNamedTensorの順序付き配列。

状態名は子Module名をドットで連結する。例: `hidden1.gamma`、`hidden2.running_var`。

## Module

### 所有権

Parameter、Buffer、子Moduleの実体は派生クラスが所有する。Module基底はraw pointerだけを登録し、破棄しない。このため登録対象はModule自身より長く生存しなければならない。

Moduleはコピー・ムーブとも禁止され、登録済みpointerの不正化を防ぐ。

### 登録API

| API | 対象 |
| --- | --- |
| `registerParameter` | Optimizer更新および保存の対象 |
| `registerBuffer` | 保存対象だがOptimizer更新対象外 |
| `registerModule` | 再帰的な状態列挙とモード伝播の対象 |

名前は空でなく、ドットを含まず、同一Module内のParameter、Buffer、子Module全体で重複してはならない。null登録と自分自身の子登録は禁止。

子Module登録時には親の現在モードを子へ同期する。

### 状態列挙

- `namedParameters`: ローカルParameterの後に、登録順の子Module Parameterを再帰列挙
- `namedBuffers`: ローカルBufferの後に、登録順の子Module Bufferを再帰列挙
- `getParams`: namedParametersからTensor pointerだけを返す
- `stateDict`: 全Parameterを先に、全Bufferを後に連結

### 学習・評価モード

初期状態は学習モード。`setTraining(bool)`は自身を更新し、全登録済み子Moduleへ再帰伝播する。`isTraining()`は現在モードを返す。

Function自体がModuleではない場合、派生Moduleがforward時にモードを同期する。現在のLayerHiddenはBatchNormへ同期し、評価時はDropoutを迂回する。

### 勾配と状態

`zero_grads`は全Parameterのgradを0へ設定する。Bufferは対象外。`reset_state`は現在no-opで、状態リセット用の拡張点。

### forward

派生クラスが実装する純粋virtual API。入力はshared Tensor配列、戻り値は単一Tensorである。Functionの複数出力APIとは別のModuleレベル規約。

## LayerConv2D

Conv2D用のweightとbiasを所有し、それぞれ`weight`、`bias`としてParameter登録する。constructor引数はinput/output channels、固定input height/width、正方形kernel、stride、padding。`outputChannels()`、`outputHeight()`、`outputWidth()`で出力geometryを取得できる。

`init(std::mt19937&)`はfan-inを`kernelSize^2 * inputChannels`としてweightをHe正規分布、biasを0で初期化する。forwardは単一入力を所有するConv2D Functionへ渡す。

## Model

Moduleを継承し、stateDictのファイル保存と読込を提供する。

### 保存形式 version 1

native binaryで次を順に書く。

```text
8 bytes magic = AICPPMDL
uint32 version = 1
uint32 entryCount

entryごと:
  uint32 nameLength
  nameLength bytes name
  int32 rows
  int32 cols
  rows * cols個のfloat
```

最大状態名長は4096。Tensor値はDeviceからHostへ転送して保存する。ParameterとBufferを保存し、勾配、計算グラフ、Optimizer状態は保存しない。

### loadの整合性検査

magic、version、entry数、名前、重複、shapeを厳密に検査する。現在のモデルが期待する全状態とファイル内容が一致しなければ失敗するため、状態項目を追加したモデルは旧ファイルと互換でない。

ファイル形式はnative endianおよびnative float表現であり、異なるarchitecture間の可搬性は保証しない。
