#!/usr/bin/env python3
"""Prepare a separate blinded n=40 validation supplement after four runs exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.future_runs import FUTURE_CONFIGURATIONS, audit_future_run

EXPECTED_LABELS = tuple(row["label"] for row in FUTURE_CONFIGURATIONS)
VERSION = "promise_switch_validation_supplement_v1_n40"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-run", action="append", required=True, metavar="LABEL=RUN_DIR",
        help="repeat exactly once for each of the four future configuration labels",
    )
    result.add_argument("--seed", required=True, type=int)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--provenance-output", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if len(args.source_run) != 4:
        raise SystemExit("exactly four --source-run arguments are required")
    sources = dict(_source(value) for value in args.source_run)
    if tuple(sources) != EXPECTED_LABELS:
        raise SystemExit("sources must use the four exact labels in the documented order")
    if args.output.exists() or args.provenance_output.exists():
        raise SystemExit("refusing to overwrite validation supplement artifacts")
    selected: list[dict[str, Any]] = []
    provenance_items: dict[str, dict[str, str]] = {}
    source_provenance = []
    rng = random.Random(args.seed)
    for label, run_dir in sources.items():
        try:
            resolved = run_dir.resolve()
            manifest, expected = audit_future_run(label, resolved)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        path = resolved / "message_events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        eligible = [row for row in rows if _eligible(row)]
        if len(eligible) < 10:
            raise SystemExit(f"{label} has only {len(eligible)} mechanically eligible messages")
        for row in rng.sample(sorted(eligible, key=_stable_key), 10):
            raw = str(row["raw_message"])
            blind_id = "PSVS2-" + hashlib.sha256(
                    f"{args.seed}|{label}|{_stable_key(row)}".encode()
                ).hexdigest()[:16]
            selected.append({
                "blind_id": blind_id,
                "message": raw,
            })
            provenance_items[blind_id] = {
                "configuration_label": label,
                "run_id": str(manifest["run_id"]),
                "trial_id": str(row.get("trial_id", "")),
                "message_event_id": str(row.get("message_event_id", "")),
                "message_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            }
        source_provenance.append({
            "configuration_label": label, "run_path": str(resolved),
            "run_id": str(manifest["run_id"]), "model_id": expected["model_id"],
            "model_alias": expected["model_alias"], "provider": expected["provider"],
            "reasoning_mode": expected["reasoning_mode"],
            "manifest_sha256": hashlib.sha256((resolved / "manifest.json").read_bytes()).hexdigest(),
            "message_events_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION, "n": 40,
        "per_configuration": 10,
        "items": selected,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(json.dumps({
        "version": VERSION, "private": True, "seed": args.seed,
        "frozen_n130_untouched": True, "sources": source_provenance,
        "items_by_blind_id": provenance_items,
    }, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


def _source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--source must be LABEL=JSONL")
    return label, Path(path)


def _eligible(row: dict[str, Any]) -> bool:
    return (
        row.get("message_recorded") is True
        and isinstance(row.get("raw_message"), str)
        and bool(row["raw_message"].strip())
    )


def _stable_key(row: dict[str, Any]) -> str:
    return str(row.get("trial_id") or row.get("item_id") or "")


if __name__ == "__main__":
    raise SystemExit(main())
