#!/usr/bin/env python3
"""Local-only converters for structured Japanese conversation-summary SFT.

Source corpora are never copied into the repository.  The output keeps source
IDs, revisions, and converter version so a model manifest can be audited.
"""
import argparse
import hashlib
import json
from pathlib import Path

SECTIONS = (
    "FACTS", "DECISIONS", "CONSTRAINTS", "OPEN_QUESTIONS",
    "ACTIVE_TASKS", "CORRECTIONS",
)
VERSION = "summary-sft-v3"


def structured(values):
    chunks = []
    for name in SECTIONS:
        items = [str(x).strip() for x in values.get(name, []) if str(x).strip()]
        chunks.append(f"[{name}]\n" + "\n".join(f"- {x}" for x in items or ["なし"]))
    return "\n".join(chunks)


def prompt(payload):
    return (
        "[Current Input]\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n[Goal]\n既存要約を逐次更新し、明示的訂正時だけ古い値を置換する"
        "\n[Current Task]\n192 tokenを目標に固定6セクションで要約する"
        "\n[Critical Constraints]\n数値、否定、訂正、期限、未解決事項、タスク状態を保持する\n"
    )


def flatten(value, prefix=""):
    out = []
    if isinstance(value, dict):
        for key in sorted(value):
            out.extend(flatten(value[key], f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten(item, prefix))
    elif value not in (None, "", False):
        out.append(f"{prefix}={value}")
    return out


def load_records(path):
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return value
        for key in ("dialogues", "data", "items"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def jmultiwoz(args):
    records = load_records(args.input)
    output = []
    excluded = {}
    for index, dialogue in enumerate(records):
        source_id = str(dialogue.get("dialogue_id", dialogue.get("id", index)))
        turns = dialogue.get("turns", dialogue.get("dialogue", []))
        utterances = []
        states = []
        for turn in turns:
            if isinstance(turn, str):
                utterances.append(turn)
                continue
            text = turn.get("utterance", turn.get("text", ""))
            if text:
                utterances.append(text)
            for key in ("dialogue_state", "state", "frames"):
                if key in turn:
                    states.extend(flatten(turn[key]))
        goal = dialogue.get("goal", dialogue.get("goals", {}))
        facts = utterances[-4:]
        constraints = list(dict.fromkeys(flatten(goal) + states))
        corrections = [x for x in utterances if "ではなく" in x or "変更" in x]
        values = {
            "FACTS": facts,
            "DECISIONS": constraints[-2:],
            "CONSTRAINTS": constraints,
            "OPEN_QUESTIONS": [x for x in utterances if "？" in x or "?" in x],
            "ACTIVE_TASKS": [x for x in flatten(goal) if x],
            "CORRECTIONS": corrections,
        }
        item = {
            "id": f"jmultiwoz-{args.split}-{source_id}", "mode": "SUMMARIZE",
            "prompt": prompt({"previous_summary": "", "turns": utterances}),
            "output": structured(values), "sft_profile": "agent",
            "split": args.split, "generation_seed": args.seed,
            "source_name": "JMultiWOZ", "source_id": source_id,
            "source_revision": args.revision, "source_license": "CC BY-SA 4.0",
            "converter_version": VERSION,
        }
        if len(item["prompt"]) + len(item["output"]) > args.max_codepoints:
            excluded["over_max_codepoints"] = excluded.get("over_max_codepoints", 0) + 1
        else:
            output.append(item)
    write(args.output, output, args, excluded, "JMultiWOZ")


def livedoor_records(path, excluded):
    if path.is_file():
        for index, record in enumerate(load_records(path)):
            title = str(record.get("title", "")).strip()
            body = str(record.get("content", record.get("body", ""))).strip()
            if not title or not body:
                excluded["malformed_article"] = excluded.get("malformed_article", 0) + 1
                continue
            source_id = str(record.get("url", record.get("id", index)))
            yield source_id, title, body, record.get("category", "")
        return
    for article in sorted(path.rglob("*.txt")):
        lines = article.read_text(encoding="utf-8").splitlines()
        if len(lines) < 4:
            excluded["malformed_article"] = excluded.get("malformed_article", 0) + 1
            continue
        yield (article.relative_to(path).as_posix(), lines[2].strip(),
               "\n".join(lines[3:]).strip(), article.parent.name)


def livedoor(args):
    output = []
    excluded = {}
    for source_id, title, body, category in livedoor_records(args.input, excluded):
        bucket = int.from_bytes(hashlib.sha256(f"{args.seed}:{source_id}".encode()).digest()[:8], "big") % 100
        if bucket >= args.mix_percent:
            continue
        item = {
            "id": f"livedoor-{args.split}-{hashlib.sha256(source_id.encode()).hexdigest()[:16]}",
            "mode": "SUMMARIZE", "prompt": prompt({"article": body}),
            "output": structured({"FACTS": [title]}), "sft_profile": "agent",
            "split": args.split, "generation_seed": args.seed,
            "source_name": "llm-book/livedoor-news-corpus", "source_id": source_id,
            "source_revision": args.revision, "source_license": "CC BY-ND 2.1 JP",
            "source_category": category, "converter_version": VERSION,
            "weak_supervision": "title-as-facts", "mix_percent": args.mix_percent,
            "rights_review_required": True,
        }
        if len(item["prompt"]) + len(item["output"]) > args.max_codepoints:
            excluded["over_max_codepoints"] = excluded.get("over_max_codepoints", 0) + 1
        else:
            output.append(item)
    write(args.output, output, args, excluded, "llm-book/livedoor-news-corpus")


def write(path, items, args, excluded, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "format": "ai_cpp_summary_sft_conversion", "version": 1,
        "converter_version": VERSION, "source_name": source,
        "source_revision": args.revision, "split": args.split,
        "generation_seed": args.seed, "accepted": len(items),
        "excluded": excluded,
    }
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def stable_pick(items, count, seed):
    ranked = sorted(items, key=lambda x: hashlib.sha256(
        f"{seed}:{x['id']}".encode()).digest())
    if len(ranked) < count:
        raise ValueError(f"requested {count} examples, only {len(ranked)} available")
    return ranked[:count]


def mix(args):
    base = read_jsonl(args.base)
    external = read_jsonl(args.jmultiwoz)
    other = [x for x in base if x.get("mode") != "SUMMARIZE"]
    synthetic = [x for x in base if x.get("mode") == "SUMMARIZE"]
    total = len(synthetic)
    j_count = round(total * 0.70)
    mixed_summary = stable_pick(external, j_count, args.seed)
    mixed_summary += stable_pick(synthetic, total - j_count, args.seed)
    if args.livedoor:
        weak = read_jsonl(args.livedoor)
        weak_count = round(total * args.livedoor_percent / 100)
        mixed_summary += stable_pick(weak, weak_count, args.seed)
    items = other + mixed_summary
    items.sort(key=lambda x: hashlib.sha256(f"{args.seed}:{x['id']}".encode()).digest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "format": "ai_cpp_summary_sft_mix", "version": 1,
        "converter_version": VERSION, "generation_seed": args.seed,
        "summary_mix": {"JMultiWOZ": j_count, "synthetic": total - j_count,
                        "livedoor": len(mixed_summary) - total},
        "livedoor_percent": args.livedoor_percent if args.livedoor else 0,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="source", required=True)
    for name in ("jmultiwoz", "livedoor"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--split", choices=("train", "validation"), required=True)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--max-codepoints", type=int, default=900)
    sub.choices["livedoor"].add_argument("--mix-percent", type=int, choices=(5, 10), required=True)
    command = sub.add_parser("mix")
    command.add_argument("--base", type=Path, required=True)
    command.add_argument("--jmultiwoz", type=Path, required=True)
    command.add_argument("--livedoor", type=Path)
    command.add_argument("--livedoor-percent", type=int, choices=(5, 10), default=5)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.source == "mix":
        mix(args)
    else:
        (jmultiwoz if args.source == "jmultiwoz" else livedoor)(args)


if __name__ == "__main__":
    main()
