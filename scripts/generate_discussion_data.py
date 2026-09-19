#!/usr/bin/env python3
"""Generate grouped, deterministic limited-comparison evaluation and SFT data."""

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

VERSION = "discussion-synthetic-v1"
CATEGORIES = (
    "no_feasible", "one_feasible", "priority", "missing", "correction",
    "different_subject_or_time", "conflict", "proposal", "unsupported_user",
    "out_of_scope",
)


def ev(identifier, origin="document"):
    return {"id": identifier, "text": f"資料記述 {identifier}",
            "source_id": "provided-document", "start": 0, "end": 8,
            "origin": origin, "content_verified": False}


def cl(identifier, subject, attribute, value, unit, evidence_id,
       time="current", stance="asserted"):
    return {"id": identifier, "speaker": "source", "subject": subject,
            "attribute": attribute, "time": time, "condition": "default",
            "value": value, "unit": unit, "stance": stance,
            "evidence_ids": [evidence_id]}


def base(scenario_id, delta):
    a_price, b_price = 30000 + delta, 50000 + delta
    evidence = [ev(name) for name in
                ("a-price", "a-memory", "a-time", "b-price", "b-memory", "b-time")]
    evidence += [ev("budget", "user"), ev("memory-limit", "user")]
    return {
        "scenario_id": scenario_id, "topic": "A案とB案の限定比較",
        "objective": "提供資料と必須条件だけで選択する",
        "options": [{"id": "A", "label": "A案"}, {"id": "B", "label": "B案"}],
        "evidence": evidence,
        "claims": [cl("ca-price", "A", "price", a_price, "JPY", "a-price"),
                   cl("ca-memory", "A", "memory", 4, "GB", "a-memory"),
                   cl("ca-time", "A", "time", 2, "s", "a-time"),
                   cl("cb-price", "B", "price", b_price, "JPY", "b-price"),
                   cl("cb-memory", "B", "memory", 2, "GB", "b-memory"),
                   cl("cb-time", "B", "time", 1, "s", "b-time")],
        "constraints": [
            {"id": "budget", "attribute": "price", "op": "<=",
             "value": 40000 + delta, "unit": "JPY", "required": True,
             "evidence_ids": ["budget"]},
            {"id": "memory-limit", "attribute": "memory", "op": "<=",
             "value": 3, "unit": "GB", "required": True,
             "evidence_ids": ["memory-limit"]}],
        "requested_attributes": ["price", "memory"], "priorities": []}


def expected(status, selected=None):
    return {"status": status, "selected_option": selected}


def scenario(category, group, variant):
    sid = f"{category}-{group}-v{variant}"
    data = base(sid, group * 100 + variant * 10)
    turns, answers = [data], []
    if category == "no_feasible":
        answers = [expected("no_feasible_option")]
    elif category == "one_feasible":
        data["constraints"][1]["value"] = 4
        answers = [expected("supported", "A")]
    elif category == "priority":
        data["constraints"][0]["value"] = 60000 + group * 100 + variant * 10
        data["constraints"][1]["value"] = 5
        data["requested_attributes"].append("time")
        data["priorities"] = [{"attribute": "time" if variant == 0 else "price",
                                "direction": "min"}]
        answers = [expected("supported", "B" if variant == 0 else "A")]
    elif category == "missing":
        data["claims"] = [claim for claim in data["claims"]
                          if claim["id"] != ("cb-memory" if variant == 0 else "ca-price")]
        answers = [expected("insufficient_evidence")]
    elif category == "correction":
        correction = ev(f"correction-{group}-{variant}", "user")
        if variant == 0:
            answers = [expected("no_feasible_option")]
            update = {"id": f"update-{group}-{variant}",
                      "kind": "correct_constraint",
                      "target_id": "budget",
                      "value": 60000 + group * 100,
                      "evidence": correction}
            answers.append(expected("supported", "B"))
        else:
            data["constraints"][0]["value"] = 60000 + group * 100
            data["constraints"][1]["value"] = 5
            data["requested_attributes"].append("time")
            data["priorities"] = [{"attribute": "time", "direction": "min"}]
            answers = [expected("supported", "B")]
            update = {"id": f"update-{group}-{variant}",
                      "kind": "set_priorities", "target_id": "priorities",
                      "value": [{"attribute": "price", "direction": "min"}],
                      "evidence": correction}
            answers.append(expected("supported", "A"))
        turns.append({"scenario_id": sid, "updates": [update]})
    elif category == "different_subject_or_time":
        extra = ev(f"old-or-other-{group}-{variant}")
        data["evidence"].append(extra)
        if variant == 0:
            data["claims"].append(cl(f"old-{group}", "A", "price", 90000,
                                     "JPY", extra["id"], time="2025-01"))
        else:
            data["claims"].append(cl(f"other-{group}", "C", "price", 90000,
                                     "JPY", extra["id"]))
        answers = [expected("no_feasible_option")]
    elif category == "conflict":
        extra = ev(f"conflict-{group}-{variant}")
        data["evidence"].append(extra)
        data["claims"].append(cl(f"conflict-{group}-{variant}", "A", "price",
                                 35000 + group, "JPY", extra["id"]))
        answers = [expected("conflicting_evidence")]
    elif category == "proposal":
        target = "ca-price" if variant == 0 else "cb-memory"
        next(claim for claim in data["claims"] if claim["id"] == target)["stance"] = "proposed"
        answers = [expected("insufficient_evidence")]
    elif category == "unsupported_user":
        target = "a-price" if variant == 0 else "b-memory"
        next(item for item in data["evidence"] if item["id"] == target)["origin"] = "user"
        answers = [expected("insufficient_evidence")]
    elif category == "out_of_scope":
        data["requested_attributes"].append("battery")
        answers = [expected("insufficient_evidence")]
    all_evidence = ["a-price", "budget", "a-memory", "memory-limit",
                    "b-price", "b-memory"]
    if category in ("missing", "proposal", "unsupported_user"):
        missing_b_memory = (category == "missing") == (variant == 0)
        evidence_ids = (["a-price", "budget", "a-memory", "memory-limit",
                         "b-price"] if missing_b_memory else
                        ["a-memory", "memory-limit", "b-price", "budget",
                         "b-memory"])
    elif category == "conflict":
        evidence_ids = ["a-price", f"conflict-{group}-{variant}"]
    elif category == "priority" or (category == "correction" and variant == 1):
        evidence_ids = ["a-price", "budget", "a-memory", "memory-limit",
                        "a-time", "b-price", "b-memory", "b-time"]
    else:
        evidence_ids = all_evidence
    for answer in answers:
        answer["evidence_ids"] = list(evidence_ids)
    if category == "correction" and variant == 0:
        answers[1]["evidence_ids"] = [
            "a-price", f"correction-{group}-{variant}",
            "a-memory", "memory-limit", "b-price", "b-memory"]
    return {"scenario_group": f"{category}-{group}", "scenario_id": sid,
            "category": category, "turns": turns, "expected": answers}


def category_splits(category, count, seed):
    """Stratify by scenario group while keeping both paraphrases together."""
    groups = list(range(count))
    groups.sort(key=lambda group: hashlib.sha256(
        f"{seed}:{category}-{group}".encode()).digest())
    train_end = max(1, int(count * 0.70))
    validation_end = max(train_end + 1, int(count * 0.85))
    return {group: ("train" if rank < train_end else
                    "validation" if rank < validation_end else "test")
            for rank, group in enumerate(groups)}


def compact_prompt(turn):
    """Serialize the decision table without repeated schema field names."""
    origins = {item["id"]: item.get("origin", "user")
               for item in turn.get("evidence", [])}
    facts = []
    for claim in turn.get("claims", []):
        evidence_id = claim.get("evidence_ids", ["?"])[0]
        facts.append("{}:{}.{}={:g}{}@{}/{}[{}]#{}:{}".format(
            claim["id"], claim["subject"], claim["attribute"],
            claim["value"], claim["unit"], claim.get("time", "current"),
            claim.get("condition", "default"), claim.get("stance", "asserted"),
            evidence_id, origins.get(evidence_id, "?")))
    constraints = ["{}:{}{}{:g}{}#{}".format(
        item["id"], item["attribute"], item["op"], item["value"],
        item["unit"], item.get("evidence_ids", ["?"])[0])
        for item in turn.get("constraints", [])]
    priorities = ["{}:{}".format(item["attribute"], item["direction"])
                  for item in turn.get("priorities", [])]
    updates = ["{}:{}:{}={}".format(item["id"], item["kind"],
               item["target_id"], item.get("value", ""))
               for item in turn.get("updates", [])]
    return "\n".join(("[Current Input]", "scenario=" + turn["scenario_id"],
        "facts=" + ";".join(facts), "constraints=" + ";".join(constraints),
        "priorities=" + ";".join(priorities),
        "requested=" + ",".join(turn.get("requested_attributes", [])),
        "updates=" + ";".join(updates), "[Goal]",
        "Return grounded decision JSON", "[Current Task]",
        "Compare only supplied facts"))

def decision_output(answer):
    status = answer["status"]
    conclusion = {"supported": "条件に基づき選択できます。",
                  "no_feasible_option": "必須条件を満たす案はありません。",
                  "conflicting_evidence": "同じ対象・時点・条件の根拠が対立しています。",
                  "insufficient_evidence": "根拠または優先順位が不足しています。"}[status]
    return {"status": status, "conclusion": conclusion,
            "selected_option": answer["selected_option"],
            "evidence_ids": answer["evidence_ids"], "open_questions": []}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--groups-per-category", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in ("train", "validation", "test")}
    for category in CATEGORIES:
        allocation = category_splits(category, args.groups_per_category,
                                     args.seed)
        for group in range(args.groups_per_category):
            for variant in range(2):
                item = scenario(category, group, variant)
                records[allocation[group]].append(item)
    rng = random.Random(args.seed)
    hashes = {}
    for split, items in records.items():
        rng.shuffle(items)
        eval_path = args.output / f"{split}.jsonl"
        sft_path = args.output / f"{split}.sft.jsonl"
        with eval_path.open("w", encoding="utf-8") as evaluation, \
             sft_path.open("w", encoding="utf-8") as sft:
            for item in items:
                evaluation.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                for turn_index, (turn, answer) in enumerate(zip(item["turns"], item["expected"])):
                    prompt = compact_prompt(copy.deepcopy(turn))
                    sft.write(json.dumps({
                        "id": item["scenario_id"] + "-turn-" + str(turn_index), "mode": "FINAL",
                        "prompt": prompt, "output": json.dumps(decision_output(answer), ensure_ascii=False),
                        "sft_profile": "agent", "split": split,
                        "generation_seed": args.seed, "generator": VERSION,
                        "source_name": "deterministic-comparison-table",
                        "source_id": item["scenario_group"], "source_revision": VERSION,
                        "converter_version": VERSION}, ensure_ascii=False) + "\n")
        hashes[split] = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    manifest = {"format": VERSION, "seed": args.seed,
                "split_unit": "scenario_group", "categories": CATEGORIES,
                "split_strategy": "per-category deterministic 70/10/20 group ranking",
                "counts": {key: len(value) for key, value in records.items()},
                "sha256": hashes,
                "teacher": "deterministic tables and comparison rules; no external LLM"}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
