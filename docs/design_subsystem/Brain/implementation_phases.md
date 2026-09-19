# Brain System 初期実装（Phase 1〜8）

## 実装範囲

`detail/README.md` のフェーズ順に、共通型、Core State、SQLite Memory、Rule Policy、Rule Planner、Execution、Post-process、External AI Adapter、Recognition境界を実装した。

- Phase 1: `include/brain/common/`
- Phase 2: `input/`, `world/`, `goal/`, `constraint/`, `logging/`
- Phase 3: `memory/`, `policy/`
- Phase 4: `planning/`（決定的RuleBasedPlannerとPlanValidator）
- Phase 5: `execution/`（依存関係、resource lock、timeout、cancel、emergency stop）
- Phase 6: `postprocess/`（Goal、Memory、Policy統計、LearningSample）
- Phase 7: `external/`（非blocking poll、timeout、cancel、決定的Fake）
- Phase 8: `preprocess/`（Fake object/ASR、rule context）とTransformer Planner境界

`BrainSystem` は上記モジュールを所有し、入力からMock Control結果の記憶までを接続する。

## 意図的な制限

- `TransformerPlanner` はモデルbundle未ロードを明示的な失敗として返すStubであり、学習済みモデルではない。
- Object DetectionとASRは決定的Fakeであり、実認識精度を主張しない。
- Context Recognitionは `goal:<type>` だけを扱う最小rule実装である。
- Condition式は初期の限定語彙だけを決定的に評価する。一般自然言語の真偽判定は行わない。
- SQLiteの`content_json`相当列は初期実装ではSemanticの`text`属性を保存する。任意Attributeの完全JSON round-tripは後続課題である。

## 検証

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DAI_CPP_BUILD_MODEL=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Brain固有テストは `brain_common_check`, `brain_phase2_check`, `brain_phase3_4_check`, `brain_phase5_8_check` である。
