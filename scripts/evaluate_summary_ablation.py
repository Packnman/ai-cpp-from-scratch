#!/usr/bin/env python3
"""Aggregate three-seed summary ablations and enforce livedoor promotion gates."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = (
    "answer_accuracy", "macro_f1", "prompt_tokens", "compression_ratio",
    "structure_failure_rate", "summary_omission_rate", "context_overflow_rate",
    "validation_loss", "bits_per_byte", "retention_1", "retention_3", "retention_5",
)
VARIANTS = ("rule", "core", "livedoor_5", "livedoor_10")


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL: one aggregate record per variant and seed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [key for key in REQUIRED if key not in row]
        if missing or "category_f1" not in row:
            raise ValueError(f"incomplete metric row: missing {missing or ['category_f1']}")
        groups[row["variant"]].append(row)
    for variant in VARIANTS:
        seeds = {row["seed"] for row in groups[variant]}
        if len(groups[variant]) != 3 or len(seeds) != 3:
            raise ValueError(f"{variant} must contain exactly three distinct seeds")

    report = {"format": "ai_cpp_summary_ablation", "version": 1,
              "variants": {}, "promotion": {}}
    for variant in VARIANTS:
        rows = groups[variant]
        categories = sorted(set.intersection(
            *(set(row["category_f1"]) for row in rows)))
        report["variants"][variant] = {
            **{key: mean(rows, key) for key in REQUIRED},
            "category_f1": {
                category: statistics.fmean(float(row["category_f1"][category])
                                           for row in rows)
                for category in categories},
        }

    baseline = report["variants"]["core"]
    for variant in ("livedoor_5", "livedoor_10"):
        candidate = report["variants"][variant]
        answer_delta = candidate["answer_accuracy"] - baseline["answer_accuracy"]
        macro_delta = candidate["macro_f1"] - baseline["macro_f1"]
        shared = set(baseline["category_f1"]) & set(candidate["category_f1"])
        category_deltas = {
            key: candidate["category_f1"][key] - baseline["category_f1"][key]
            for key in sorted(shared)}
        prompt_reduction = ((baseline["prompt_tokens"] - candidate["prompt_tokens"])
                            / baseline["prompt_tokens"] if baseline["prompt_tokens"] else 0.0)
        quality_floor = (answer_delta >= -0.01 and macro_delta >= -0.01 and
                         all(delta >= -0.02 for delta in category_deltas.values()))
        benefit = (answer_delta >= 0.01 or macro_delta >= 0.01 or
                   (answer_delta >= 0 and macro_delta >= 0 and prompt_reduction >= 0.05))
        report["promotion"][variant] = {
            "accepted": quality_floor and benefit,
            "answer_accuracy_delta": answer_delta,
            "macro_f1_delta": macro_delta,
            "category_f1_delta": category_deltas,
            "prompt_token_reduction": prompt_reduction,
            "quality_floor_passed": quality_floor,
            "benefit_passed": benefit,
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
