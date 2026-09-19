# プロジェクトのフォルダ構成

## 配置規則

| フォルダ | 役割 | Git管理方針 |
|---|---|---|
| `evaluation/` | 再現可能な評価条件と小さな基準結果 | 条件、README、採用した小容量結果を管理 |
| `experiments/` | 比較実験に必要な固定fixture | 再生成可能な大規模データは追加しない |
| `include/` | 外部から利用するC++公開API | `ai/`と`brain/`をサブシステム境界とする |
| `src/` | C++実装 | `agent/`, `brain/`, `model/`, `ner/`に分類 |
| `output/` | PDFなどの生成成果物 | Git対象外。必要な成果物だけ別途明示して配布 |
| `scripts/` | データ変換、学習、評価、実行pipeline | 利用者が直接実行する入口を置く |
| `tests/` | 自動テストと小さなfixture | `agent/`, `brain/`, `model/`, `ner/`, `fixtures/`に分類 |
| `third_party/` | repository内へ固定した外部依存 | upstream単位で配置し、独自実装を混在させない |
| `train/` | 学習・評価CLIの`main` | 共有ロジックは`src/`へ置く |
| `tmp/` | 一時生成物 | Git対象外。現在は常設しない |

従来の`model/src/`は`src/model/`へ統合した。公開model APIは引き続き
`include/ai/model/`にあり、学習済みbundleは`models/`（Git対象外）へ保存する。

## 現在の主要構成

```text
include/
├── ai/
│   ├── agent/
│   ├── model/
│   └── ner/
└── brain/

src/
├── agent/
├── brain/
├── model/
└── ner/

tests/
├── agent/
├── brain/
├── model/
├── ner/
└── fixtures/
```

## 生成物の扱い

- build treeは`build/`またはworkspace外の`/tmp/`へ作成する。
- Pythonの`__pycache__`、`tmp/`、`output/`はGitへ追加しない。
- 学習データ本文とmodel bundleはGitへ追加せず、取得・変換・学習手順を管理する。
- 評価値を残す場合は、測定条件と再実行コマンドを`evaluation/`または`docs/validation/`へ記録する。
