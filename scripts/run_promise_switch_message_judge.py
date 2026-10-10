#!/usr/bin/env python3
"""Run or recover blinded Promise Switch semantic-judge annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.client import ProviderAgnosticClient
from promise_switch.coding.semantic_v2 import validate_v2_judge_route, verify_frozen_codebook
from promise_switch.codebook import CODEBOOK_VERSION, SEMANTIC_FIELDS, MessageCode, validate_annotation_record
from promise_switch.execution.transport import OpenAICompatibleTransport
from promise_switch.providers import create_model_config


DEFAULT_CODING_DIR = ROOT / "promise_switch" / "coding"
DEFAULT_FROZEN_KEY = ROOT / "notes/promise_switch/semantic_coding/promise_switch_message_codebook_v1/private_full_semantic_key_n2738.json"
PAID_ACK = "I AUTHORIZE PAID PROMISE SWITCH SEMANTIC JUDGE EXECUTION"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sample", type=Path, required=True)
    result.add_argument("--private-sample-key", type=Path, required=True)
    result.add_argument("--judge-prompt", type=Path, default=DEFAULT_CODING_DIR / "judge_prompt_v1.txt")
    result.add_argument("--annotation-schema", type=Path, default=DEFAULT_CODING_DIR / "annotation_schema_v1.json")
    result.add_argument("--provider", required=True)
    result.add_argument("--model", required=True, help="model alias from promise_switch.providers")
    result.add_argument("--reasoning-mode", choices=("on", "off", "unavailable"), required=True)
    result.add_argument("--temperature", type=float, required=True)
    result.add_argument("--max-output-tokens", type=int, required=True)
    result.add_argument("--judge-id", required=True)
    result.add_argument("--judge-config-id", required=True)
    result.add_argument(
        "--assignment", choices=("all", "gpt-oss-origin", "non-gpt-oss-origin"), required=True,
    )
    result.add_argument("--annotations-out", type=Path, required=True)
    result.add_argument("--provenance-out", type=Path, required=True)
    result.add_argument("--base-provenance", type=Path)
    result.add_argument("--existing-annotations", type=Path)
    result.add_argument("--consolidated-annotations-out", type=Path)
    result.add_argument("--expected-target-count", type=int)
    result.add_argument("--max-attempts-per-item", type=int, default=3)
    result.add_argument("--timeout-seconds", type=float, default=120.0)
    result.add_argument("--verify-reasoning", action="store_true")
    result.add_argument("--frozen-codebook-commit", default="9e8f59e")
    result.add_argument(
        "--frozen-codebook-key", type=Path, default=DEFAULT_FROZEN_KEY,
        help="local linkage artifact used only by the optional private coding workflow",
    )
    result.add_argument(
        "--semantic-v2-manifest", type=Path,
        help="Bind this run to the new-only judge inputs and routing policy in a v2 manifest",
    )
    result.add_argument("--live-submit", action="store_true")
    result.add_argument("--acknowledge-paid-provider-call")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_gpt_oss_origin(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(field, ""))
        for field in ("source_model_configuration", "source_run_name")
    ).lower()
    return "gpt_oss" in text or "gpt-oss" in text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def selected_by_assignment(source: dict[str, Any], assignment: str) -> bool:
    if assignment == "all":
        return True
    return is_gpt_oss_origin(source) == (assignment == "gpt-oss-origin")


def annotation_record(
    parsed: dict[str, Any], *, judge_id: str, judge_config_id: str, config: Any,
) -> dict[str, Any]:
    result = {
        "item_id": parsed["item_id"],
        "message": parsed["message"],
        "codebook_version": CODEBOOK_VERSION,
        **{field: parsed[field] for field in SEMANTIC_FIELDS},
        "judge_id": judge_id,
        "judge_provider": config.provider,
        "judge_model_id": config.model_id,
        "judge_reasoning_mode": config.reasoning_mode,
        "judge_config_id": judge_config_id,
    }
    reasoning_effort = config.provider_request_kwargs.get("reasoning_effort")
    if reasoning_effort is not None:
        result["judge_reasoning_effort"] = reasoning_effort
    else:
        result["judge_reasoning_configuration"] = config.provider_request_kwargs
    return result


def validate_existing_annotations(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if not item_id or item_id in by_id:
            raise ValueError("existing annotations require unique nonempty item_id values")
        if row.get("codebook_version") != CODEBOOK_VERSION:
            raise ValueError("existing annotation codebook_version mismatch")
        MessageCode.from_mapping(row)
        by_id[item_id] = row
    return by_id


def validate_output(
    raw: str, *, item: dict[str, str], schema_validator: Draft202012Validator,
) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("judge output must be one JSON object")
    schema_errors = list(schema_validator.iter_errors(parsed))
    if schema_errors:
        raise ValueError("JSON Schema: " + "; ".join(error.message for error in schema_errors))
    validate_annotation_record(parsed)
    if parsed["item_id"] != item["item_id"]:
        raise ValueError("item_id mismatch")
    if parsed["message"] != item["message"]:
        raise ValueError("message mismatch")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.live_submit:
        raise SystemExit("semantic judge execution requires --live-submit")
    if args.acknowledge_paid_provider_call != PAID_ACK:
        raise SystemExit(
            "semantic judge execution requires --acknowledge-paid-provider-call "
            f"'{PAID_ACK}'"
        )
    if args.semantic_v2_manifest:
        manifest = json.loads(args.semantic_v2_manifest.read_text(encoding="utf-8"))
        new_items = manifest.get("new_items", {})
        if (
            Path(str(new_items.get("judge_sample_path", ""))).resolve() != args.sample.resolve()
            or Path(str(new_items.get("judge_private_key_path", ""))).resolve()
            != args.private_sample_key.resolve()
        ):
            raise SystemExit("semantic-v2 judge inputs do not match the versioned manifest")
        try:
            observed_hashes = verify_frozen_codebook(
                codebook_path=DEFAULT_CODING_DIR / "codebook_v1.md",
                validator_path=ROOT / "promise_switch/codebook.py",
                judge_prompt_path=args.judge_prompt,
                annotation_schema_path=args.annotation_schema,
                frozen_key_path=args.frozen_codebook_key,
            )
            recorded_hashes = dict(manifest.get("frozen_codebook", {}))
            recorded_hashes.pop("verified", None)
            if recorded_hashes != observed_hashes:
                raise ValueError("semantic-v2 manifest codebook hashes do not match")
            validate_v2_judge_route(
                manifest, judge_provider=args.provider, judge_model=args.model,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if args.max_attempts_per_item < 1:
        raise SystemExit("--max-attempts-per-item must be positive")
    if bool(args.base_provenance) != bool(args.existing_annotations):
        raise SystemExit("--base-provenance and --existing-annotations must be supplied together")
    if args.consolidated_annotations_out and not args.existing_annotations:
        raise SystemExit("--consolidated-annotations-out requires recovery inputs")
    output_paths = [args.annotations_out, args.provenance_out]
    if args.consolidated_annotations_out:
        output_paths.append(args.consolidated_annotations_out)
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise SystemExit("Refusing to overwrite existing judge artifacts: " + ", ".join(existing_outputs))

    with args.sample.open(newline="", encoding="utf-8-sig") as handle:
        sample = list(csv.DictReader(handle))
    private_key = json.loads(args.private_sample_key.read_text(encoding="utf-8"))
    key_rows = private_key["items"]
    source_by_id = {row["item_id"]: row for row in key_rows}
    sample_ids = [row["item_id"] for row in sample]
    if len(source_by_id) != len(key_rows) or set(sample_ids) != set(source_by_id):
        raise SystemExit("sample/private-key item mismatch")
    for row in sample:
        expected = source_by_id[row["item_id"]]["message_sha256"]
        actual = sha256_bytes(row["message"].encode("utf-8"))
        if actual != expected:
            raise SystemExit("private-key message hash mismatch")

    assignment_ids = {
        row["item_id"] for row in sample
        if selected_by_assignment(source_by_id[row["item_id"]], args.assignment)
    }
    existing_rows: list[dict[str, Any]] = []
    existing_by_id: dict[str, dict[str, Any]] = {}
    base_provenance: dict[str, Any] | None = None
    if args.base_provenance:
        base_provenance = json.loads(args.base_provenance.read_text(encoding="utf-8"))
        existing_rows = read_jsonl(args.existing_annotations)
        existing_by_id = validate_existing_annotations(existing_rows)
        base_assigned = {
            row["item_id"] for row in base_provenance["assignments"]
            if row["assigned_judge"] == base_provenance["judge_id"]
        }
        if not base_assigned <= assignment_ids:
            raise SystemExit("base provenance includes items outside the source-family filter")
        if not set(existing_by_id) <= base_assigned:
            raise SystemExit("existing annotation is outside the base assignment")
        target_ids = base_assigned - set(existing_by_id)
    else:
        target_ids = assignment_ids
    targets = [row for row in sample if row["item_id"] in target_ids]
    if args.expected_target_count is not None and len(targets) != args.expected_target_count:
        raise SystemExit(
            f"expected {args.expected_target_count} target items, found {len(targets)}"
        )

    prompt_template = args.judge_prompt.read_text(encoding="utf-8")
    schema = json.loads(args.annotation_schema.read_text(encoding="utf-8"))
    schema_validator = Draft202012Validator(schema)
    config = create_model_config(
        provider=args.provider, model_alias=args.model, reasoning_mode=args.reasoning_mode,
    )
    client = ProviderAgnosticClient(
        config, OpenAICompatibleTransport(timeout_seconds=args.timeout_seconds),
    )
    if args.verify_reasoning:
        probe = client.complete(
            messages=[{
                "role": "user",
                "content": "Non-scientific configuration probe. Reply with exactly OK.",
            }],
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
        )
        if probe.status != "success" or not probe.error_metadata.get("reasoning_signal_present"):
            raise SystemExit("reasoning-on probe did not produce a mechanical reasoning signal")

    annotations: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "artifact_version": (
            "promise_switch_judge_recovery_provenance_v1"
            if base_provenance else "promise_switch_judge_provenance_v1"
        ),
        "private": True,
        "mode": "recovery" if base_provenance else "initial",
        "codebook_version": CODEBOOK_VERSION,
        "frozen_codebook_commit": args.frozen_codebook_commit,
        "judge_id": args.judge_id,
        "judge_config_id": args.judge_config_id,
        "judge_provider": config.provider,
        "judge_model_id": config.model_id,
        "judge_reasoning_mode": config.reasoning_mode,
        "judge_reasoning_request": config.provider_request_kwargs,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "timeout_seconds": args.timeout_seconds,
        "retry_policy": {
            "maximum_total_attempts_per_item": args.max_attempts_per_item,
            "retryable": [
                "api_or_infrastructure_failure", "structural_or_schema_invalid_output",
            ],
            "no_retry_for_valid_annotation_based_on_labels": True,
        },
        "judge_prompt_sha256": sha256_bytes(args.judge_prompt.read_bytes()),
        "schema_sha256": sha256_bytes(args.annotation_schema.read_bytes()),
        "sample_sha256": sha256_bytes(args.sample.read_bytes()),
        "private_sample_key_sha256": sha256_bytes(args.private_sample_key.read_bytes()),
        "base_provenance_sha256": (
            sha256_bytes(args.base_provenance.read_bytes()) if args.base_provenance else None
        ),
        "existing_annotations_sha256": (
            sha256_bytes(args.existing_annotations.read_bytes())
            if args.existing_annotations else None
        ),
        "start_timestamp_utc": utc_now(),
        "end_timestamp_utc": None,
        "counts": {
            "sample": len(sample),
            "source_family_assignment": len(assignment_ids),
            "existing_valid": len(existing_by_id),
            "targeted": len(targets),
            "attempted": 0,
            "completed": 0,
            "schema_invalid_failed": 0,
            "infrastructure_failed": 0,
            "failed": 0,
        },
        "assignments": [{
            "item_id": row["item_id"],
            "source_model_configuration": source_by_id[row["item_id"]][
                "source_model_configuration"
            ],
            "source_run_name": source_by_id[row["item_id"]]["source_run_name"],
            "assigned_judge": args.judge_id,
            "actual_judge_model": config.model_id,
            "judge_provider": config.provider,
            "judge_reasoning_mode": config.reasoning_mode,
            "status": "pending",
        } for row in targets],
        "attempts": [],
    }
    args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.annotations_out, annotations)
    atomic_json(args.provenance_out, provenance)

    for index, item in enumerate(targets, 1):
        item_id = item["item_id"]
        prompt = prompt_template.replace("{{ITEM_ID}}", item_id).replace(
            "{{MESSAGE}}", item["message"]
        )
        completed = False
        final_kind = "infrastructure"
        final_reason = "no attempt completed"
        provenance["counts"]["attempted"] += 1
        for attempt_number in range(1, args.max_attempts_per_item + 1):
            attempt_started = utc_now()
            response = client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
            )
            attempt = {
                "item_id": item_id,
                "attempt_number": attempt_number,
                "start_timestamp_utc": attempt_started,
                "end_timestamp_utc": utc_now(),
                "status": response.status,
                "http_status": response.http_status,
                "provider_response_id": response.provider_response_id,
                "provider_request_id": response.provider_request_id,
                "finish_reason": response.finish_reason,
                "latency_seconds": response.latency_seconds,
                "usage": response.usage,
                "reasoning_signal_present": response.error_metadata.get(
                    "reasoning_signal_present"
                ),
                "reasoning_signal_fields": response.error_metadata.get(
                    "reasoning_signal_fields", []
                ),
            }
            parsed = None
            if response.status != "success":
                error = "API/infrastructure failure"
                failure_kind = "infrastructure"
                attempt["error_metadata"] = response.error_metadata
            elif not response.error_metadata.get("reasoning_signal_present"):
                error = "successful response lacked mechanical reasoning signal"
                failure_kind = "infrastructure"
            else:
                try:
                    parsed = validate_output(
                        response.raw_response, item=item, schema_validator=schema_validator,
                    )
                    error = None
                    failure_kind = None
                except Exception as exc:
                    error = str(exc)
                    failure_kind = "schema_or_structural"
            if error is None:
                annotations.append(annotation_record(
                    parsed, judge_id=args.judge_id,
                    judge_config_id=args.judge_config_id, config=config,
                ))
                attempt["validation_status"] = "valid"
                attempt["response_sha256"] = sha256_bytes(
                    response.raw_response.encode("utf-8")
                )
                completed = True
            else:
                attempt["validation_status"] = "invalid"
                attempt["validation_error"] = error
                attempt["raw_judge_response"] = response.raw_response
                final_reason = error
                final_kind = failure_kind
            provenance["attempts"].append(attempt)
            provenance["counts"]["completed"] = len(annotations)
            atomic_jsonl(args.annotations_out, annotations)
            atomic_json(args.provenance_out, provenance)
            if completed:
                break
            if attempt_number < args.max_attempts_per_item:
                time.sleep(min(2 ** attempt_number, 8))
        assignment = next(
            row for row in provenance["assignments"] if row["item_id"] == item_id
        )
        assignment["attempt_count"] = attempt_number
        if completed:
            assignment["status"] = "valid"
        else:
            assignment["status"] = "failed"
            assignment["failure_kind"] = final_kind
            assignment["failure_reason"] = final_reason
            provenance["counts"]["failed"] += 1
            counter = (
                "schema_invalid_failed"
                if final_kind == "schema_or_structural" else "infrastructure_failed"
            )
            provenance["counts"][counter] += 1
        atomic_json(args.provenance_out, provenance)
        print(
            f"progress {index}/{len(targets)} valid={len(annotations)} "
            f"failed={provenance['counts']['failed']}",
            flush=True,
        )

    provenance["end_timestamp_utc"] = utc_now()
    provenance["counts"]["completed"] = len(annotations)
    provenance["counts"]["failed"] = len(targets) - len(annotations)
    atomic_jsonl(args.annotations_out, annotations)
    atomic_json(args.provenance_out, provenance)
    if args.consolidated_annotations_out:
        combined = {**existing_by_id}
        for row in annotations:
            if row["item_id"] in combined:
                raise RuntimeError("recovery produced a duplicate existing annotation")
            combined[row["item_id"]] = row
        consolidated = [combined[item_id] for item_id in sample_ids if item_id in combined]
        args.consolidated_annotations_out.parent.mkdir(parents=True, exist_ok=True)
        atomic_jsonl(args.consolidated_annotations_out, consolidated)
    print(json.dumps(provenance["counts"], indent=2, sort_keys=True))
    return 0 if not provenance["counts"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
