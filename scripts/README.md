# Scripts

このディレクトリには、現在のソースとCLIで実行確認できる入口だけを置く。
生成データ、学習bundle、build成果物はGitへ追加しない。

## 通常buildと実行

- `run.sh`: CPU既定構成をbuildし、rule backendの `agent_cli` を起動する。

## Robot Simulation

- `simulation/run.sh test`: MuJoCo/Actuatorの単体・結合テスト
- `simulation/run.sh sanitizer`: ASan/UBSan付きテスト
- `simulation/run.sh headless`: humanoid modelのheadless実行
- `simulation/run.sh smoke`: 全4段階MJCF modelのload/step確認
- `simulation/run.sh viewer`: GLFW viewerのbuildと起動

詳細と環境変数は [simulation/README.md](simulation/README.md) を参照する。

## Brainテスト

- `brain/run_tests.sh spec`: 仕様書単位の12テストとBrain結合テスト
- `brain/run_tests.sh all`: 仕様テストと既存Phase回帰テスト
- `brain/run_tests.sh sanitizer`: Brain全テストをASan/UBSan付きで実行

詳細は [brain/README.md](brain/README.md) を参照する。

## v3モデルpipeline

設定の正本は `agent_pipeline_env.sh`。既定値はbalanced tokenizer、
`data/jawiki-20260901-shards`、`models/agent_v3_pretrain_jawiki`、
`models/agent_v3_chat_sft` を指す。

1. `run_00_status.sh`: 必要物と次の段階を表示
2. `run_01_build.sh`: CUDA model buildとテスト
3. `run_02_tokenizer.sh`: Jawiki＋会話データからbalanced tokenizerを生成
4. `run_03_shard_jawiki.sh`: Jawiki trainを再現可能なshardへ変換
5. `run_03_pretrain.sh`: shard事前学習。checkpointがあれば自動再開
6. `run_04_generate_sft.sh`: chatまたはagent SFTデータを生成
7. `run_05_sft.sh`: 完了済みv3 pretrainから別bundleへSFT
8. `run_06_validate.sh`: validation splitを評価
9. `run_07_agent.sh`: hybrid backendを起動

既存bundleは上書きしない。smokeは出力先とshard数を明示して実行する。

```sh
AI_CPP_PRETRAIN_MODEL=models/agent_v3_pretrain_smoke \
AI_CPP_MAX_SHARDS=1 ./scripts/run_03_pretrain.sh
```

agent profileではJMultiWOZのローカル固定revisionが必要になる。

```sh
AI_CPP_SFT_PROFILE=agent \
AI_CPP_JMULTIWOZ_DIR=/path/to/jmultiwoz \
AI_CPP_JMULTIWOZ_REVISION=FIXED_REVISION \
./scripts/run_04_generate_sft.sh
```

## 独立pipelineと変換

- `fetch_ner_wikipedia.sh`, `convert_ner_wikipedia.py`, `run_ner_pipeline.sh`:
  固定revisionのNERデータ取得、変換、学習、評価
- `convert_summary_sft.py`: JMultiWOZ/livedoorの要約SFT変換と混合
- `generate_discussion_data.py`, `mix_discussion_sft.py`:
  限定比較データ生成と既存SFTへの決定的混合
- `evaluate_summary_ablation.py`: 3 seedの要約ablation集計と採用判定

詳細なライセンス、split、評価条件は `docs/japanese_ner.md`、
`docs/conversation_summary.md`、`docs/discussion_evaluation.md` を参照する。
