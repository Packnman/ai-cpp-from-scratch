#!/usr/bin/env python3
"""Finalize a pinned WikiExtractor JSONL corpus into deterministic train/validation/test JSONL."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs(path: Path):
    if path.is_file():
        yield path
    else:
        yield from sorted(p for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted", type=Path,
                        help="WikiExtractor JSONL file or directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--dump-date", required=True, help="YYYYMMDD")
    parser.add_argument("--dump-url", required=True)
    parser.add_argument("--dump-sha256", required=True)
    parser.add_argument("--extractor-revision", required=True)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-documents", type=int, default=0)
    args = parser.parse_args()

    actual_hash = sha256(args.dump_file)
    if actual_hash.lower() != args.dump_sha256.lower():
        raise SystemExit(
            f"dump SHA-256 mismatch: expected={args.dump_sha256} actual={actual_hash}")
    if args.min_chars <= 0 or args.max_documents < 0:
        raise SystemExit("min-chars must be positive and max-documents nonnegative")

    args.output.mkdir(parents=True, exist_ok=True)
    streams = {
        name: (args.output / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    counts = {name: 0 for name in streams}
    seen = set()
    accepted = 0
    try:
        for path in inputs(args.extracted):
            with path.open(encoding="utf-8") as source:
                for line in source:
                    value = json.loads(line)
                    page_id = str(value["id"])
                    title = str(value.get("title", ""))
                    text = str(value["text"]).replace("\r\n", "\n").replace("\r", "\n")
                    if text.lstrip().upper().startswith("#REDIRECT"):
                        continue
                    paragraphs = [
                        " ".join(paragraph.split())
                        for paragraph in text.split("\n\n")
                    ]
                    for paragraph_index, paragraph in enumerate(paragraphs):
                        if len(paragraph) < args.min_chars:
                            continue
                        key = f"{page_id}:{paragraph_index}"
                        identifier = int.from_bytes(
                            hashlib.sha256(key.encode()).digest()[:8], "big") & ((1 << 63) - 1)
                        if identifier in seen:
                            raise SystemExit(f"derived document ID collision: {key}")
                        seen.add(identifier)
                        bucket = hashlib.sha256(page_id.encode()).digest()[0] % 20
                        split = "train" if bucket < 18 else (
                            "validation" if bucket == 18 else "test")
                        streams[split].write(json.dumps(
                            {"id": identifier, "source_page_id": page_id,
                             "title": title,
                             "url": str(value.get("url", "")),
                             "text": paragraph},
                            ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[split] += 1
                        accepted += 1
                        if args.max_documents and accepted >= args.max_documents:
                            break
                    if args.max_documents and accepted >= args.max_documents:
                        break
            if args.max_documents and accepted >= args.max_documents:
                break
    finally:
        for stream in streams.values():
            stream.close()

    if any(count == 0 for count in counts.values()):
        raise SystemExit(f"every split must be nonempty: {counts}")
    metadata = {
        "format": "ai_cpp_jawiki_text_v1",
        "dump_date": args.dump_date,
        "dump_url": args.dump_url,
        "dump_file": args.dump_file.name,
        "dump_sha256": actual_hash,
        "extractor": "WikiExtractor JSON output",
        "extractor_revision": args.extractor_revision,
        "conditions": {
            "input": "WikiExtractor JSONL; main namespace and redirect filtering expected",
            "redirects": "lines whose extracted text begins with #REDIRECT are excluded",
            "unit": "blank-line paragraph, whitespace collapsed",
            "minimum_characters": args.min_chars,
            "split": "SHA-256(page_id)[0] modulo 20: 0-17 train, 18 validation, 19 test",
            "max_documents": args.max_documents,
        },
        "counts": counts,
        "source_terms": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en",
        "content_license_notice": "Preserve page attribution from the dump/extractor and review Wikimedia licensing before redistributing weights.",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
