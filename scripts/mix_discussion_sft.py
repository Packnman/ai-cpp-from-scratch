#!/usr/bin/env python3
"""Append deterministic discussion examples to an existing agent SFT split."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--discussion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    base = read(args.base)
    discussion = read(args.discussion)
    identifiers = [item["id"] for item in base + discussion]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate SFT id")
    items = base + discussion
    items.sort(key=lambda item: hashlib.sha256(
        f"{args.seed}:{item['id']}".encode()).digest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "format": "ai_cpp_discussion_sft_mix_v1",
        "seed": args.seed,
        "base": {"path": str(args.base), "count": len(base),
                 "sha256": hashlib.sha256(args.base.read_bytes()).hexdigest()},
        "discussion": {"path": str(args.discussion),
                       "count": len(discussion),
                       "sha256": hashlib.sha256(
                           args.discussion.read_bytes()).hexdigest()},
        "output_count": len(items),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
