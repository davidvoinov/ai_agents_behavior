#!/usr/bin/env python3
"""Prepare a new semantic-v2 corpus without altering frozen v1 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.coding.semantic_v2 import (
    FROZEN_BASE_CORPUS_SIZE, GPT_LUNA_ROUTING_POLICIES, SEMANTIC_CORPUS_VERSION,
    verify_frozen_codebook,
)
from promise_switch.future_runs import FUTURE_CONFIGURATIONS, audit_future_run

DEFAULT_BASE = ROOT / "notes/promise_switch/semantic_coding/promise_switch_message_codebook_v1/final_full_semantic_annotations_n2738.jsonl"
DEFAULT_BASE_KEY = ROOT / "notes/promise_switch/semantic_coding/promise_switch_message_codebook_v1/private_full_semantic_key_n2738.json"
DEFAULT_CODEBOOK = ROOT / "promise_switch/coding/codebook_v1.md"
DEFAULT_VALIDATOR = ROOT / "promise_switch/codebook.py"
DEFAULT_PROMPT = ROOT / "promise_switch/coding/judge_prompt_v1.txt"
DEFAULT_SCHEMA = ROOT / "promise_switch/coding/annotation_schema_v1.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-annotations", type=Path, default=DEFAULT_BASE)
    result.add_argument("--base-key", type=Path, default=DEFAULT_BASE_KEY)
    result.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    result.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    result.add_argument("--judge-prompt", type=Path, default=DEFAULT_PROMPT)
    result.add_argument("--annotation-schema", type=Path, default=DEFAULT_SCHEMA)
    result.add_argument(
        "--source-run", action="append", required=True, metavar="LABEL=RUN_DIR",
        help="completed audited future run; repeat in registered configuration order",
    )
    result.add_argument(
        "--gpt-luna-routing-policy", required=True, choices=GPT_LUNA_ROUTING_POLICIES,
    )
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite versioned corpus directory: {args.output_dir}")
    try:
        codebook_hashes = verify_frozen_codebook(
            codebook_path=args.codebook, validator_path=args.validator,
            judge_prompt_path=args.judge_prompt,
            annotation_schema_path=args.annotation_schema,
            frozen_key_path=args.base_key,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    base_bytes = args.base_annotations.read_bytes()
    base_rows = [json.loads(line) for line in base_bytes.decode("utf-8").splitlines()
                 if line.strip()]
    if len(base_rows) != FROZEN_BASE_CORPUS_SIZE:
        raise SystemExit(f"frozen base must contain exactly {FROZEN_BASE_CORPUS_SIZE} annotations")
    if len({row.get("item_id") for row in base_rows}) != len(base_rows):
        raise SystemExit("frozen base contains duplicate annotation identities")
    new_items: list[dict[str, Any]] = []
    source_runs = []
    seen_labels: set[str] = set()
    order = {row["label"]: index for index, row in enumerate(FUTURE_CONFIGURATIONS)}
    observed_order: list[int] = []
    for value in args.source_run:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or label in seen_labels:
            raise SystemExit("--source-run must be unique LABEL=RUN_DIR")
        seen_labels.add(label)
        try:
            run_dir = Path(raw_path).resolve()
            manifest, expected = audit_future_run(label, run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        observed_order.append(order[label])
        messages_path = run_dir / "message_events.jsonl"
        source_runs.append({
            "label": label, "run_path": str(run_dir), "run_id": manifest["run_id"],
            "manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
            "message_events_sha256": hashlib.sha256(messages_path.read_bytes()).hexdigest(),
            "model_id": expected["model_id"], "model_alias": expected["model_alias"],
            "provider": expected["provider"], "reasoning_mode": expected["reasoning_mode"],
            "source_model_family": expected["family"],
        })
        for row in _read_jsonl(messages_path):
            if not _eligible(row):
                continue
            trial_id = str(row["trial_id"])
            new_items.append({
                "item_id": "PSV2-" + hashlib.sha256(
                    f"{manifest['run_id']}|{trial_id}".encode()
                ).hexdigest()[:20],
                "trial_id": trial_id,
                "source_configuration": label,
                "source_run_name": str(manifest["run_id"]),
                "source_message_event_id": row.get("message_event_id"),
                "source_model_family": expected["family"],
                "raw_message": row["raw_message"],
                "message_sha256": hashlib.sha256(row["raw_message"].encode("utf-8")).hexdigest(),
                "annotation_status": "pending_new_item_only",
            })
    if observed_order != sorted(observed_order):
        raise SystemExit("source runs must follow registered future configuration order")
    if not new_items:
        raise SystemExit("audited source runs contain no eligible voluntary messages")
    if len({row["item_id"] for row in new_items}) != len(new_items):
        raise SystemExit("duplicate semantic-v2 item identity")
    args.output_dir.mkdir(parents=True)
    items_path = args.output_dir / "new_items_pending_coding.jsonl"
    items_path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(new_items, key=lambda row: row["item_id"])
    ))
    sample_path = args.output_dir / "new_items_blinded.csv"
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("item_id", "message"))
        writer.writeheader()
        for row in sorted(new_items, key=lambda item: item["item_id"]):
            writer.writerow({"item_id": row["item_id"], "message": row["raw_message"]})
    key_path = args.output_dir / "private_new_items_key.json"
    key_path.write_text(json.dumps({
        "artifact_version": "promise_switch_semantic_v2_private_key_v1",
        "corpus_version": SEMANTIC_CORPUS_VERSION,
        "items": [{
            "item_id": row["item_id"],
            "message_sha256": row["message_sha256"],
            "source_model_configuration": row["source_configuration"],
            "source_model_family": row["source_model_family"],
            "source_run_name": row["source_run_name"],
            "source_trial_id": row["trial_id"],
            "source_message_event_id": row["source_message_event_id"],
        } for row in sorted(new_items, key=lambda item: item["item_id"])],
    }, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "corpus_version": SEMANTIC_CORPUS_VERSION,
        "frozen_codebook_changed": False,
        "frozen_base": {
            "path": str(args.base_annotations.resolve()),
            "n": FROZEN_BASE_CORPUS_SIZE,
            "sha256": hashlib.sha256(base_bytes).hexdigest(),
            "private_key_path": str(args.base_key.resolve()),
            "private_key_sha256": hashlib.sha256(args.base_key.read_bytes()).hexdigest(),
            "reuse_policy": "reference existing annotations unchanged; never re-code",
        },
        "frozen_codebook": {"verified": True, **codebook_hashes},
        "new_items": {
            "path": str(items_path.resolve()), "n": len(new_items),
            "coding": "new items only",
            "judge_sample_path": str(sample_path.resolve()),
            "judge_private_key_path": str(key_path.resolve()),
        },
        "combined_n": FROZEN_BASE_CORPUS_SIZE + len(new_items),
        "gpt_luna_routing_policy": args.gpt_luna_routing_policy,
        "source_model_families": sorted({row["source_model_family"] for row in new_items}),
        "source_runs": source_runs,
    }
    (args.output_dir / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("message_recorded") is True
        and isinstance(row.get("trial_id"), str)
        and isinstance(row.get("raw_message"), str)
    )


if __name__ == "__main__":
    raise SystemExit(main())
