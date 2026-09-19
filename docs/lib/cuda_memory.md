# CUDA メモリ管理

`lib/include/cuda_memory.h` は内部用の共通管理層。学習・評価・対話生成の `cuStorage<float>` / `cuStorage<int32_t>` と Mask / Permute / Softmax の補助領域に適用する。Tensor・モデルの公開操作および bundle v1 の保存形式は変更しない。

## 設定と保持方針

`AI_CPP_CUDA_MEMORY_POOL=1`（既定）は、デバイスごとに専用の `cudaMemPool_t` を必要時に作成し、`cudaMallocFromPoolAsync` / `cudaFreeAsync` を既定 stream に投入する。演算の stream、入力検証、既存の同期と CPU 転送は維持する。release threshold は `UINT64_MAX`。全 GPU 容量の事前確保は行わず、必要に応じて拡張する。通常の同期・バッチ終了では明示的に trim しない。

`AI_CPP_CUDA_MEMORY_POOL=0` は従来の `cudaMalloc` / `cudaFree`。設定はプロセス内の最初の非ゼロ確保時に固定する。不正な設定値は例外。非対応デバイス・ドライバーでは理由を stderr に記録して従来方式へフォールバックする。通常の CUDA エラーやメモリ不足ではフォールバックせず例外を伝える。

`scripts/run_train.sh` と `scripts/run_validation.sh` は `MEMORY_POOL=1` を既定とし、上記環境変数として子プロセスに渡す。例：

```sh
MEMORY_POOL=0 bash scripts/run_train.sh
MEMORY_POOL=1 MODE=chat bash scripts/run_validation.sh
```

変更は次に起動するプロセスから適用される。稼働中の学習の設定や状態には影響しない。

## 所有権と終了

`cu_memory::Buffer` はコピー禁止の RAII バッファ。確保元デバイスのプールを共有所有し、解放では元のデバイスへ切り替え、呼出元のデバイスへ戻す。`cuStorage` を共有するビューと計算グラフの寿命は従来どおり。最後の参照がなくなった時点でバッファを返却する。ゼロ要素では GPU 確保を行わない。

レジストリは実行中のプールを保持する。CUDA ランタイムをレジストリより先に初期化し、終了時にプールを先に破棄する。バッファがレジストリより長生きする場合も共有所有でプールを保持する。プール破棄前にデバイスを同期し、未解放割当がないことを確認する。デストラクタは例外を送出せず、失敗は stderr に記録する。同期失敗や未解放割当がある場合はプールを強制破棄しない。明示的な `Buffer::release()` の失敗は例外。

新しい stream を導入する場合は、この管理層の既定 stream 契約を再検討する必要がある。

## 指標・内部 API

最初のデバイス確保時に stderr に `allocator=pool` または `allocator=legacy` を記録する。学習開始の JSON にも `cuda_allocator` を記録する。各 split の指標には同期後の `pool_used_bytes` と `pool_reserved_bytes` を追加し、従来の `sampled_device_used_bytes` / `baseline_device_used_bytes` も保持する。従来方式の pool 指標は 0。

**予約済みメモリは計算に使用中とは限らない。** `pool_used_bytes` は未返却割当、`pool_reserved_bytes` は再利用待ち領域を含む専用プールの予約量。デバイス全体の使用量には他プロセスや cuBLAS 等も含まれる。予約量の一定性や高速化率は保証しない。

- `cu_memory::statistics()`：現在のデバイスを同期して統計を取得する。未初期化時はゼロを返し、設定は固定しない。
- `cu_memory::trimUnused()`：現在のデバイスを同期し、未使用領域を明示解放する。内部検証用であり、通常の処理ループでは呼ばない。

## 検証・性能比較

```sh
cmake -S . -B /tmp/ai-cpp-memory-pool-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ai-cpp-memory-pool-build -j2
AI_CPP_CUDA_MEMORY_POOL=1 ctest --test-dir /tmp/ai-cpp-memory-pool-build --output-on-failure -j1
AI_CPP_CUDA_MEMORY_POOL=0 ctest --test-dir /tmp/ai-cpp-memory-pool-build --output-on-failure -j1
```

`cuda_memory_check` は float / 整数、ゼロ要素、サイズ変更、ビュー寿命、例外時解放、同期後の予約保持、ウォームアップ後の予約量安定、明示 trim を確認する。ポインタ一致は要求しない。`cuda_memory_compare` は独立プロセスで同一 seed・重み・入力の Transformer を実行し、初期重み・logits・loss・保持した過去グラフの勾配・Adam 更新後重みを絶対誤差 1e-6 以内で比較する。保存・再読込・追加学習・単独評価・生成は既存 `conversation_check` が検証する。

GPU の学習が終了してから、固定した同一 bundle とデータで以下を実行する。各方式3回、バッチ32、3バッチのウォームアップ後に10バッチを測定し、秒・tokens/s・予約量を JSON で出力する。入力モデルとデータには書き込まない。少なくとも416ウィンドウが必要。スクリプトは各実行前に他の GPU 計算プロセスがある場合に中止する。測定中も別の学習を開始しないこと。

```sh
BUILD_DIR=/tmp/ai-cpp-memory-pool-build bash scripts/run_memory_benchmark.sh /path/to/stable-bundle data/conversation/train.jsonl > /tmp/memory-benchmark.jsonl
```

方式の参考：[NVIDIA stream-ordered allocator](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)。

### 今回の検証結果（2026-09-12）

RTX 3060 Ti、CUDA Toolkit 12.8、Release の独立ビルド `/tmp/ai-cpp-memory-pool-build` で、プール有効・無効の各17テストが全件成功した。数値比較の8,389値は最大絶対差0で一致。固定サイズの再利用、予約保持、trim、ビュー・過去グラフの寿命、例外時解放、保存・再読込・追加学習・評価・生成を確認した。CUDA 終了順序の回帰チェックも成功した。

既存 `main_train`（PID 146275）が稼働中のため、性能比較は未実施。正当性テストの所要時間は性能比較値として扱わない。性能測定スクリプトが当該プロセスを検出して測定開始前に中止することを確認した。学習は停止・変更していない。非対応 GPU / ドライバーのフォールバックと複数 GPU 間の切替は、この単一 GPU 環境では実機未検証。
