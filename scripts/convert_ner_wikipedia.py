#!/usr/bin/env python3
"""Validate Stockmark NER spans and create page-safe deterministic JSONL splits."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

LABELS = ["人名", "法人名", "政治的組織名", "その他の組織名", "地名", "施設名", "製品名", "イベント名"]
REVISION = "5e525c2132c0e87cce890b8b92639a0cf357c1f3"
SOURCE_SHA256 = "5796effb85c473aec5b85784545349ed42e408bb0fecdb78f82bc107d53edbe3"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def byte_offset(text, codepoint):
    return len(text[:codepoint].encode("utf-8"))


def validate(rows):
    stats = collections.Counter()
    rejected = []
    for row_index, row in enumerate(rows):
        if not isinstance(row.get("curid"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"row {row_index}: curid/text schema mismatch")
        entities = row.get("entities")
        if not isinstance(entities, list):
            raise ValueError(f"row {row_index}: entities is not a list")
        spans = []
        for entity in entities:
            if entity.get("type") not in LABELS:
                raise ValueError(f"row {row_index}: unknown label {entity.get('type')!r}")
            span = entity.get("span")
            if not (isinstance(span, list) and len(span) == 2 and
                    all(isinstance(value, int) for value in span)):
                raise ValueError(f"row {row_index}: invalid span")
            start, end = span
            # Upstream spans are Unicode-code-point, half-open positions.
            if not (0 <= start < end <= len(row["text"])):
                raise ValueError(f"row {row_index}: out-of-range span {span}")
            if row["text"][start:end] != entity.get("name"):
                raise ValueError(f"row {row_index}: span/name mismatch {entity!r}")
            spans.append((start, end))
            stats[f"label:{entity['type']}"] += 1
        spans.sort()
        conflict = any(right_start < left_end for (_, left_end), (right_start, _) in zip(spans, spans[1:]))
        if conflict:
            stats["bio_unrepresentable"] += 1
            rejected.append(row_index)
        if not entities:
            stats["negative"] += 1
    return stats, set(rejected)


def components(rows):
    """Join pages that share an exact sentence, preventing duplicate leakage."""
    union = UnionFind()
    first_page = {}
    duplicates = 0
    for row in rows:
        page = row["curid"]
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if digest in first_page:
            duplicates += 1
            union.union(page, first_page[digest])
        else:
            first_page[digest] = page
        union.find(page)
    grouped = collections.defaultdict(set)
    for page in union.parent:
        grouped[union.find(page)].add(page)
    return grouped, duplicates


def assign_splits(rows, seed):
    grouped, duplicate_count = components(rows)
    page_split = {}
    for pages in grouped.values():
        key = ",".join(sorted(pages))
        value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big") % 100
        split = "train" if value < 80 else "validation" if value < 90 else "test"
        for page in pages:
            page_split[page] = split
    return page_split, duplicate_count


def chunks(row, max_chars):
    text = row["text"]
    entities = sorted(row["entities"], key=lambda entity: entity["span"])
    if len(text) <= max_chars:
        return [(0, len(text), entities)], 0
    result, split_count = [], 0
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        crossing = [entity for entity in entities if entity["span"][0] < end < entity["span"][1]]
        if crossing:
            end = max(entity["span"][1] for entity in crossing)
        if end < len(text):
            safe = [i + 1 for i in range(start, end) if text[i] in "。！？、"]
            candidate = safe[-1] if safe else end
            if not any(entity["span"][0] < candidate < entity["span"][1] for entity in entities):
                end = candidate
        if end <= start:
            raise ValueError("failed to make progress while chunking")
        local = []
        for entity in entities:
            left, right = entity["span"]
            if start <= left and right <= end:
                item = dict(entity)
                item["span"] = [left - start, right - start]
                local.append(item)
        result.append((start, end, local))
        split_count += end < len(text)
        start = end
    return result, split_count


def convert(source, output, seed, max_chars):
    raw = source.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw)
    stats, rejected = validate(rows)
    page_split, duplicate_count = assign_splits(rows, seed)
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "test")}
    rejected_handle = (output / "rejected.jsonl").open("w", encoding="utf-8")
    counts = collections.Counter()
    try:
        for index, row in enumerate(rows):
            if index in rejected:
                rejected_handle.write(json.dumps({"reason": "overlapping_or_nested_span", "row": row}, ensure_ascii=False) + "\n")
                continue
            split = page_split[row["curid"]]
            pieces, split_count = chunks(row, max_chars)
            counts["long_splits"] += split_count
            for chunk_index, (start, end, entities) in enumerate(pieces):
                text = row["text"][start:end]
                converted = []
                for entity in entities:
                    left, right = entity["span"]
                    converted.append({
                        "type": entity["type"], "surface": entity["name"],
                        "codepoint_span": [left, right],
                        "byte_span": [byte_offset(text, left), byte_offset(text, right)],
                    })
                item = {"curid": row["curid"], "chunk": chunk_index, "text": text,
                        "entities": converted,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                handles[split].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[f"examples:{split}"] += 1
                counts[f"negative:{split}"] += not converted
    finally:
        for handle in handles.values():
            handle.close()
        rejected_handle.close()
    manifest = {
        "format": "ai_cpp_ner_jsonl_v1", "source": str(source),
        "upstream": "https://github.com/stockmarkteam/ner-wikipedia-dataset",
        "revision": REVISION, "expected_source_sha256": SOURCE_SHA256,
        "actual_source_sha256": actual_sha, "seed": seed,
        "split_policy": "SHA-256(seed, connected component of curid plus exact duplicate text), 80/10/10",
        "span_policy": "upstream Unicode code-point half-open; output includes code-point and UTF-8 byte half-open",
        "labels": LABELS, "max_codepoints": max_chars,
        "counts": dict(sorted((stats + counts).items())),
        "exact_duplicate_rows": duplicate_count,
        "bio_conflict_policy": "write complete original rows to rejected.jsonl; never silently discard",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-chars", type=int, default=256)
    args = parser.parse_args()
    if args.max_chars < 8:
        parser.error("--max-chars must be at least 8")
    print(json.dumps(convert(args.source, args.output, args.seed, args.max_chars), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
