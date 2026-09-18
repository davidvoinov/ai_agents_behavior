#!/usr/bin/env python3
"""Validate new semantic-v2 labels and create an explicitly activated merged corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.codebook import (
    CODEBOOK_VERSION, SEMANTIC_FIELDS, MessageCode, validate_annotation_record,
)
from promise_switch.coding.semantic_v2 import SEMANTIC_CORPUS_VERSION
from promise_switch.future_runs import audit_future_run


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared-manifest", required=True, type=Path)
    result.add_argument("--new-annotations", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite semantic-v2 finalization: {args.output_dir}")
    manifest = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    if manifest.get("corpus_version") != SEMANTIC_CORPUS_VERSION:
        raise SystemExit("prepared manifest has the wrong semantic corpus version")
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list) or not source_runs:
        raise SystemExit("prepared manifest has no audited source runs")
    labels = [str(row.get("label", "")) for row in source_runs if isinstance(row, dict)]
    if len(labels) != len(source_runs) or len(labels) != len(set(labels)):
        raise SystemExit("prepared manifest source-run identities are malformed or duplicated")
    for source in source_runs:
        run_dir = Path(str(source.get("run_path", ""))).resolve()
        try:
            audited_manifest, expected = audit_future_run(str(source["label"]), run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
        observed = {
            "run_id": audited_manifest.get("run_id"),
            "model_id": expected["model_id"], "model_alias": expected["model_alias"],
            "provider": expected["provider"], "reasoning_mode": expected["reasoning_mode"],
            "source_model_family": expected["family"],
        }
        recorded = {name: source.get(name) for name in observed}
        if recorded != observed:
            raise SystemExit("prepared manifest source-run identity changed after audit")
        if _sha(run_dir / "manifest.json") != source.get("manifest_sha256"):
            raise SystemExit("audited source run manifest changed after preparation")
        if _sha(run_dir / "message_events.jsonl") != source.get("message_events_sha256"):
            raise SystemExit("audited source messages changed after preparation")
    frozen = manifest.get("frozen_base", {})
    base_path = Path(str(frozen.get("path", "")))
    base_key_path = Path(str(frozen.get("private_key_path", "")))
    if not base_path.is_file() or _sha(base_path) != frozen.get("sha256"):
        raise SystemExit("frozen annotation corpus hash mismatch")
    if not base_key_path.is_file() or _sha(base_key_path) != frozen.get("private_key_sha256"):
        raise SystemExit("frozen semantic private-key hash mismatch")
    pending_path = Path(str(manifest.get("new_items", {}).get("path", "")))
    pending = _rows(pending_path)
    annotations = _rows(args.new_annotations)
    pending_by_id = {str(row.get("item_id", "")): row for row in pending}
    annotation_ids = [str(row.get("item_id", "")) for row in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise SystemExit("duplicate semantic-v2 coding result")
    if set(annotation_ids) != set(pending_by_id):
        raise SystemExit("semantic-v2 coding is incomplete or contains unexpected items")
    canonical_annotations = []
    for row in annotations:
        canonical = {
            "item_id": row.get("item_id"), "message": row.get("message"),
            "codebook_version": row.get("codebook_version"),
            "coder_id": row.get("coder_id") or row.get("judge_id") or row.get("judge_config_id"),
            **{field: row.get(field) for field in SEMANTIC_FIELDS},
        }
        try:
            validate_annotation_record(canonical)
            MessageCode.from_mapping(canonical)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid semantic-v2 coding result: {exc}") from exc
        if row.get("codebook_version") != CODEBOOK_VERSION:
            raise SystemExit("semantic-v2 coding result uses the wrong codebook")
        expected = pending_by_id[str(row["item_id"])]
        if row.get("message") != expected.get("raw_message"):
            raise SystemExit("semantic-v2 coding result message mismatch")
        canonical_annotations.append(canonical)
    base = _rows(base_path)
    base_ids = {str(row.get("item_id", "")) for row in base}
    if base_ids.intersection(annotation_ids):
        raise SystemExit("new semantic coding collides with the frozen corpus")
    base_key = json.loads(base_key_path.read_text(encoding="utf-8"))
    new_key = json.loads(Path(str(manifest["new_items"]["judge_private_key_path"])).read_text(
        encoding="utf-8"
    ))
    key_rows = new_key.get("items")
    if not isinstance(key_rows, list) or {
        str(row.get("item_id", "")) for row in key_rows if isinstance(row, dict)
    } != set(pending_by_id) or len(key_rows) != len(pending_by_id):
        raise SystemExit("semantic-v2 pending items/private key coverage mismatch")
    key_by_id = {str(row["item_id"]): row for row in key_rows}
    for item_id, item in pending_by_id.items():
        if key_by_id[item_id].get("message_sha256") != hashlib.sha256(
            str(item.get("raw_message", "")).encode("utf-8")
        ).hexdigest():
            raise SystemExit("semantic-v2 pending message/private key hash mismatch")
    combined = [*base, *sorted(canonical_annotations, key=lambda row: str(row["item_id"]))]
    combined_key = {
        **base_key,
        "artifact_version": "promise_switch_semantic_v2_combined_private_key_v1",
        "corpus_version": SEMANTIC_CORPUS_VERSION,
        "items": [*base_key.get("items", []), *new_key.get("items", [])],
        "source_runs": [
            *base_key.get("source_runs", []),
            *(str(row["run_id"]) for row in manifest.get("source_runs", [])),
        ],
        "semantic_v2_source_runs": manifest.get("source_runs", []),
    }
    if len(combined_key["items"]) != len(combined):
        raise SystemExit("combined semantic annotation/private-key coverage mismatch")
    args.output_dir.mkdir(parents=True)
    annotations_out = args.output_dir / "final_combined_semantic_annotations.jsonl"
    key_out = args.output_dir / "private_combined_semantic_key.json"
    _write_jsonl(annotations_out, combined)
    key_out.write_text(json.dumps(combined_key, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    activation = {
        "corpus_version": SEMANTIC_CORPUS_VERSION,
        "explicit_activation_required": True,
        "annotations_path": str(annotations_out.resolve()),
        "annotations_sha256": _sha(annotations_out),
        "private_key_path": str(key_out.resolve()),
        "private_key_sha256": _sha(key_out),
        "frozen_base_sha256": frozen["sha256"],
        "frozen_base_count": len(base),
        "new_item_count": len(annotations),
        "combined_count": len(combined),
        "frozen_codebook": manifest.get("frozen_codebook"),
        "gpt_luna_routing_policy": manifest.get("gpt_luna_routing_policy"),
        "source_runs": manifest.get("source_runs", []),
    }
    (args.output_dir / "semantic_v2_activation.json").write_text(
        json.dumps(activation, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
