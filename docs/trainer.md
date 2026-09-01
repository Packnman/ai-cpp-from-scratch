# trainer.h 設計仕様

対象: `include/trainer.h`

## 目的

特定のModel型とDataset型へ依存しない、教師あり学習と評価のテンプレート関数を提供する。

## TrainConfig

| フィールド | 意味 | 制約・既定値 |
| --- | --- | --- |
| `nEpochs` | 学習エポック数 | 1以上 |
| `nBatchSize` | バッチサイズ | 1以上 |
| `nSeed` | シャッフルおよびモデル構築に使うseed | 呼び出し側が設定 |
| `fLearningRate` | Optimizer学習率 | 現在はtrainer側で範囲検証しない |
| `fDropoutRate` | Dropoutで落とす確率 | 既定値0.2、Dropout構築時に検証 |

## SupervisedBatch

`spmInput`、`spmTarget`、実サンプル数`nSize`をまとめる。Datasetの`makeBatch`が返す共通形式である。

## TrainingState

エポック番号と累積更新回数を保持するための構造体。現在の`train`実装では未使用であり、checkpointや再開学習用の拡張点である。

## Dataset要件

Dataset型は次のAPIを提供する必要がある。

```cpp
std::size_t size() const;
SupervisedBatch makeBatch(
    const std::vector<std::size_t>& indices,
    std::size_t begin,
    std::size_t count
) const;
```

## Model要件

学習時は`setTraining`、`loss`、評価時は`setTraining`、`forward`を提供する必要がある。

## train

1. モデルを学習モードへ切り替える。
2. 設定と非空データセットを検証する。
3. Optimizerを`init`する。
4. エポックごとにインデックスをseed付き`std::mt19937`でshuffleする。
5. バッチごとに勾配をゼロ化する。
6. loss forward、`backward`、Optimizer更新を実行する。
7. サンプル数で重み付けした平均lossを表示する。

最終バッチはデータセット残数に合わせて縮小する。関数終了後もモデルは学習モードのままである。

## evaluate

モデルを評価モードへ切り替え、元の順序でバッチを生成する。Metricは`metric(output, target)`で正解件数を返すCallableでなければならない。全正解数をデータセットサイズで割ったfloatを返す。

評価ではOptimizer更新を行わない。ただし専用のno-grad機構はなく、Function呼び出しは通常どおりContextを生成する。各バッチの一時Tensor解放によりグラフも解放される。関数終了後もモデルは評価モードのままである。

## 内部補助

- `makeIndices`: 0からsize-1までの連番を生成
- `readScalar`: 1 × 1 TensorをGPUからCPUへ転送
- `validateConfig`: epochsとbatch sizeを検証

## countClassificationCorrect

MNISTとCIFAR-10で共有する分類精度metricである。同じ正のshapeを持つlogitsとone-hot targetをGPUからCPUへ転送し、各batch列のargmaxが一致する件数を返す。Softmaxは不要である。nullまたはshape不一致は例外となる。
