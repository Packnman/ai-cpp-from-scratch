#!/usr/bin/env python3
import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        subprocess.run(
            [str(ROOT / "scripts/generate_discussion_data.py"), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((output / "manifest.json").read_text())
        locations = defaultdict(set)
        category_by_split = {}
        for split in ("train", "validation", "test"):
            raw = (output / f"{split}.jsonl").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == manifest["sha256"][split]
            records = [json.loads(line) for line in raw.decode().splitlines()]
            category_by_split[split] = {item["category"] for item in records}
            for item in records:
                locations[item["scenario_group"]].add(split)
            for line in (output / f"{split}.sft.jsonl").read_text().splitlines():
                example = json.loads(line)
                assert example["mode"] == "FINAL"
                assert len(example["prompt"]) <= 766
        assert all(len(splits) == 1 for splits in locations.values())
        expected = set(manifest["categories"])
        assert all(categories == expected for categories in category_by_split.values())
    print("discussion_data_check: passed")


if __name__ == "__main__":
    main()
