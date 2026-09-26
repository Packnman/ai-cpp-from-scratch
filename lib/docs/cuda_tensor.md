# cuda_tensor.h 設計仕様

`Tensor`は値`_mData`と累積勾配`_mGrad`を同じN次元shapeの`cufMat`として確保し、自動微分graphの`Context`を保持する。

`Tensor(rows,cols)`は既存の2次元APIとして残り、`Tensor(shape)`で任意rankを構築できる。どちらもdataとgradはrow-majorであり、gradは構築時に0へ初期化される。

`backward()`は起点勾配を1でseedし、Contextを出力側から入力側へ逆順に実行する。各Functionのbackwardは入力とparameterの勾配へ加算するため、学習step前に`zero_grads()`が必要である。

出力TensorはContextをshared pointerで所有し、Contextは入力Tensorをshared pointer、出力Tensorをweak pointerで参照する。Function pointerは非所有であり、Functionはgraphより長く生存する必要がある。
