#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("convert_ner", ROOT / "scripts/convert_ner_wikipedia.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConversionTest(unittest.TestCase):
    def test_spans_pages_duplicates_negatives_and_conflicts(self):
        rows = [
            {"curid": "1", "text": "東京へ行く", "entities": [{"name": "東京", "span": [0, 2], "type": "地名"}]},
            {"curid": "2", "text": "東京へ行く", "entities": [{"name": "東京", "span": [0, 2], "type": "地名"}]},
            {"curid": "3", "text": "負例です", "entities": []},
            {"curid": "4", "text": "東京都", "entities": [
                {"name": "東京都", "span": [0, 3], "type": "地名"},
                {"name": "東京", "span": [0, 2], "type": "地名"}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "source.json"; output = root / "out"
            source.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            manifest = module.convert(source, output, 7, 8)
            located = {}
            for split in ("train", "validation", "test"):
                for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    item = json.loads(line); located[item["curid"]] = (split, item)
            self.assertEqual(located["1"][0], located["2"][0])
            self.assertEqual(located["1"][1]["entities"][0]["byte_span"], [0, 6])
            self.assertIn("3", located)
            self.assertEqual(manifest["counts"]["bio_unrepresentable"], 1)
            self.assertIn("overlapping_or_nested_span", (output / "rejected.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
