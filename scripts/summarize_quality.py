#!/usr/bin/env python3
"""Summarize token budgets and response validation metrics for A/B/C/D."""

import json
import sys
from pathlib import Path


def optimized_tokens(model: Path) -> int:
    result = 0
    with (model / "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("split") == "train":
                result = max(result, int(value.get("optimized_tokens", 0)))
    return result


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "experiments/quality-comparison")
    variants = {
        "A": (root / "A-current-conversation", []),
        "B": (root / "B-current-mixed", [root / "B-pretrain"]),
        "C": (root / "C-large-conversation", []),
        "D": (root / "D-large-mixed", [root / "D-pretrain"]),
    }
    rows = []
    for name, (model, earlier) in variants.items():
        validation_path = root / f"{model.name}-validation.json"
        chat_path = root / f"{model.name}-fixed-chat" / "top-k-1.log"
        if not model.is_dir() or not validation_path.is_file():
            continue
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        tokens = optimized_tokens(model) + sum(optimized_tokens(path) for path in earlier)
        chat = chat_path.read_text(encoding="utf-8") if chat_path.is_file() else ""
        rows.append((name, tokens, validation["loss"], validation["perplexity"],
                     chat.count("[system] 会話終端を検出したため、履歴をリセットしました。")))
    output = [
        "# 会話品質比較結果", "",
        "| 構成 | 実最適化token | response validation loss | response perplexity | top-k 1 reset |",
        "|---|---:|---:|---:|---:|",
    ]
    output += [
        f"| {name} | {tokens} | {loss:.6f} | {perplexity:.6f} | {resets} |"
        for name, tokens, loss, perplexity, resets in rows
    ]
    output += [
        "",
        "固定5問の回答本文とEND位置は各構成の fixed-chat 内のログを参照する。",
        "A対Bがデータ効果、A対Cがモデル規模効果である。",
        "DはBとCの両方がAより改善したと判断した場合だけ実行する。",
        "",
    ]
    (root / "summary.md").write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    main()
