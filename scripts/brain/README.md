# Brain tests

`tests/brain/` にあるC++テストを、CPUだけの独立buildで実行する。
生成物は既定でGit対象外の `build/brain-tests*` に置く。

```sh
# 12モジュール仕様＋Brain結合仕様＋Context model仕様
./scripts/brain/run_tests.sh spec

# 上記にcommon/Phase回帰テストを加えたBrain全テスト
./scripts/brain/run_tests.sh all

# Brain全テストをASan/UBSan付きで実行
./scripts/brain/run_tests.sh sanitizer
```

第2引数以降はCTestへ渡す。仕様別のPASS/SKIP出力を見る場合は次を使う。

```sh
./scripts/brain/run_tests.sh spec -V
```

build先と並列数は環境変数で変更できる。

```sh
AI_CPP_BRAIN_BUILD_DIR=/tmp/ai-cpp-brain-tests \
AI_CPP_JOBS=4 ./scripts/brain/run_tests.sh all
```
