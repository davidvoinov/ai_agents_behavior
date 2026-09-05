#!/usr/bin/env python3
"""Prepare, execute, resume, and finalize full Promise Switch semantic coding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.client import ProviderAgnosticClient
from promise_switch.codebook import (
    CODEBOOK_VERSION,
    SEMANTIC_FIELDS,
    MessageCode,
    eligible_explicit_promiser,
    validate_annotation_record,
)
from promise_switch.execution.transport import OpenAICompatibleTransport
from promise_switch.providers import create_model_config


FROZEN_CODEBOOK_COMMIT = "9e8f59eb38c32ecd89030ddb175fb7090075e56b"
PERSISTED_RUNNER_COMMIT = "45e82854aef297f8128308c31613678a6a52c32d"
VALIDATION_DIR = (
    ROOT / "notes" / "promise_switch" / "validation" / CODEBOOK_VERSION
)
OUTPUT_DIR = (
    ROOT / "notes" / "promise_switch" / "semantic_coding" / CODEBOOK_VERSION
)
CODING_DIR = ROOT / "promise_switch" / "coding"
UNIVERSE_PATH = OUTPUT_DIR / "full_semantic_universe_n2738.csv"
PRIVATE_KEY_PATH = OUTPUT_DIR / "private_full_semantic_key_n2738.json"
STATE_PATH = OUTPUT_DIR / "private_full_semantic_state.sqlite3"
ATTEMPTS_PATH = OUTPUT_DIR / "private_judge_attempts.jsonl"
DESIGNATIONS_PATH = OUTPUT_DIR / "private_final_designations.jsonl"
FINAL_PATH = OUTPUT_DIR / "final_full_semantic_annotations_n2738.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "coding_manifest.json"
MINIMAX_ATTEMPTS_PATH = OUTPUT_DIR / "private_minimax_judge4_attempts.jsonl"
FORMAT_RECOVERY_ATTEMPTS_PATH = (
    OUTPUT_DIR / "private_minimax_format_only_recovery_attempts.jsonl"
)
PROMPT_PATH = CODING_DIR / "judge_prompt_v1.txt"
SCHEMA_PATH = CODING_DIR / "annotation_schema_v1.json"
CODEBOOK_PATH = CODING_DIR / "codebook_v1.md"
PYTHON_VALIDATOR_PATH = ROOT / "promise_switch" / "codebook.py"
PRIMARY_KEY_PATH = VALIDATION_DIR / "private_sample_key_primary_n130.json"
EXPECTED_ELIGIBLE = 2738
EXPECTED_REUSED = 130
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 2048
MAX_ATTEMPTS = 3
MINIMAX_RECOVERY_IDS = (
    "VMFULL0365",
    "VMFULL0673",
    "VMFULL1333",
    "VMFULL1350",
    "VMFULL2057",
)
FORMAT_RECOVERY_IDS = ("VMFULL1333", "VMFULL1350")
FORMAT_ONLY_INSTRUCTION = (
    "Return exactly one raw JSON object. Do not use Markdown code fences, backticks, "
    "commentary, explanations, prefixes, suffixes, or any text before or after the JSON object."
)


@dataclass(frozen=True)
class JudgeSpec:
    key: str
    judge_id: str
    family: str
    provider: str
    model_alias: str
    reasoning_mode: str


JUDGES = {
    "gpt_primary": JudgeSpec(
        key="gpt_primary",
        judge_id="gpt_oss_120b_nebius_full_primary",
        family="gpt_oss",
        provider="nebius",
        model_alias="gpt_oss_120b_nebius",
        reasoning_mode="on",
    ),
    "deepseek_primary": JudgeSpec(
        key="deepseek_primary",
        judge_id="deepseek_v4_pro_nebius_full_primary",
        family="deepseek",
        provider="nebius",
        model_alias="deepseek_v4_pro_nebius_thinking_on",
        reasoning_mode="on",
    ),
    "deepseek_fallback": JudgeSpec(
        key="deepseek_fallback",
        judge_id="deepseek_v4_pro_nebius_full_fallback",
        family="deepseek",
        provider="nebius",
        model_alias="deepseek_v4_pro_nebius_thinking_on",
        reasoning_mode="on",
    ),
    "glm_fallback": JudgeSpec(
        key="glm_fallback",
        judge_id="glm_5_2_nebius_uk_full_fallback",
        family="glm",
        provider="nebius_uk",
        model_alias="glm_5_2_nebius_thinking_on",
        reasoning_mode="on",
    ),
    "minimax_recovery": JudgeSpec(
        key="minimax_recovery",
        judge_id="minimax_m3_nebius_reasoning_on_judge4",
        family="minimax",
        provider="nebius",
        model_alias="minimax_m3_nebius_thinking_on",
        reasoning_mode="on",
    ),
}

REUSED_FILES = {
    "gpt_oss_120b_nebius_reasoning_on_judge1": (
        VALIDATION_DIR
        / "gpt_oss_120b_nebius_reasoning_on_judge1_consolidated_annotations.jsonl",
        106,
        "gpt_oss",
    ),
    "deepseek_v4_pro_nebius_reasoning_on_judge2": (
        VALIDATION_DIR / "deepseek_v4_pro_nebius_reasoning_on_judge2_annotations.jsonl",
        20,
        "deepseek",
    ),
    "deepseek_v4_pro_nebius_reasoning_on_judge3": (
        VALIDATION_DIR / "deepseek_v4_pro_nebius_reasoning_on_judge3_annotations.jsonl",
        4,
        "deepseek",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare", "probe", "probe-minimax", "run", "recover-minimax",
            "recover-format-only", "finalize", "all",
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def model_family(*values: Any) -> str:
    text = " ".join(str(value) for value in values).lower().replace("-", "_")
    families = (
        ("gpt_oss", ("gpt_oss",)),
        ("deepseek", ("deepseek",)),
        ("qwen", ("qwen",)),
        ("glm", ("glm",)),
        ("minimax", ("minimax",)),
        ("kimi", ("kimi", "moonshot")),
        ("llama", ("llama",)),
    )
    matches = [family for family, markers in families if any(marker in text for marker in markers)]
    if len(matches) != 1:
        raise ValueError(f"cannot determine one model family from {values!r}")
    return matches[0]


def source_metadata(run_name: str) -> dict[str, Any]:
    manifest_path = ROOT / "runs" / "promise_switch" / run_name / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configurations = manifest.get("model_configurations", [])
    if len(configurations) != 1:
        raise ValueError(f"expected one source configuration in {manifest_path}")
    config = configurations[0]
    return {
        "source_model_configuration": config["model_alias"],
        "source_model_id": config["model_id"],
        "source_model_family": model_family(config["model_alias"], config["model_id"]),
        "source_provider": config["provider"],
        "source_reasoning_mode": config["reasoning_mode"],
    }


def build_universe() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_key = json.loads(PRIMARY_KEY_PATH.read_text(encoding="utf-8"))
    if primary_key["total_eligible"] != EXPECTED_ELIGIBLE:
        raise ValueError("primary provenance eligible total is not 2738")
    if primary_key.get("excluded_source_run") in primary_key["source_runs"]:
        raise ValueError("excluded source run is present in the frozen run list")
    rows: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    for run_name in primary_key["source_runs"]:
        metadata = source_metadata(run_name)
        path = ROOT / "runs" / "promise_switch" / run_name / "message_events.jsonl"
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                source = json.loads(line)
                raw_message = source.get("raw_message")
                parsed_message = source.get("parsed_message")
                eligible = (
                    source.get("message_recorded") is True
                    and source.get("parse_valid") is True
                    and isinstance(raw_message, str)
                    and isinstance(parsed_message, str)
                    and raw_message == parsed_message
                )
                if not eligible:
                    continue
                count += 1
                rows.append({
                    "source_trial_id": source["trial_id"],
                    "source_message_event_id": source["message_event_id"],
                    "source_record_location": str(path.relative_to(ROOT)),
                    "source_record_line": line_number,
                    "message": raw_message,
                    **metadata,
                    "source_run_name": run_name,
                })
        eligible_counts[run_name] = count
    if eligible_counts != primary_key["eligible_counts_by_configuration"]:
        raise ValueError("eligible counts do not match frozen validation provenance")
    if len(rows) != EXPECTED_ELIGIBLE:
        raise ValueError(f"expected 2738 eligible records, found {len(rows)}")
    identities = {
        (row["source_trial_id"], row["source_message_event_id"]) for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("eligible corpus contains duplicate source identities")
    rows.sort(key=lambda row: sha256_bytes(
        (row["source_trial_id"] + "\0" + row["source_message_event_id"]).encode("utf-8")
    ))
    for index, row in enumerate(rows, 1):
        row["item_id"] = f"VMFULL{index:04d}"
        row["message_sha256"] = sha256_bytes(row["message"].encode("utf-8"))
    return rows, primary_key


def public_record(item: dict[str, Any]) -> dict[str, str]:
    return {
        "item_id": item["item_id"],
        "message": item["message"],
        "codebook_version": CODEBOOK_VERSION,
    }


def universe_csv(items: list[dict[str, Any]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=("item_id", "message", "codebook_version"), lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(public_record(item) for item in items)
    return output.getvalue()


def private_key(items: list[dict[str, Any]], primary_key: dict[str, Any], csv_text: str) -> dict[str, Any]:
    private_items = [{key: value for key, value in item.items() if key != "message"} for item in items]
    return {
        "artifact_version": "promise_switch_full_semantic_key_v1",
        "private": True,
        "codebook_version": CODEBOOK_VERSION,
        "frozen_codebook_commit": FROZEN_CODEBOOK_COMMIT,
        "persisted_judge_runner_commit": PERSISTED_RUNNER_COMMIT,
        "created_at_utc": utc_now(),
        "eligible_population": len(items),
        "source_runs": primary_key["source_runs"],
        "excluded_source_run": primary_key.get("excluded_source_run"),
        "eligibility_rule": primary_key["eligibility_rule"],
        "ordering_rule": (
            "Sort by SHA-256 of source_trial_id, NUL, and source_message_event_id; "
            "assign VMFULL0001 through VMFULL2738."
        ),
        "fields_not_loaded_from_later_artifacts": [
            "condition", "Switch assignment", "final action", "behavioral outcome",
            "compliance", "promise violation",
        ],
        "file_hashes": {
            "universe_csv_sha256": sha256_bytes(csv_text.encode("utf-8")),
            "primary_validation_key_sha256": sha256_path(PRIMARY_KEY_PATH),
            "judge_prompt_sha256": sha256_path(PROMPT_PATH),
            "annotation_schema_sha256": sha256_path(SCHEMA_PATH),
            "codebook_markdown_sha256": sha256_path(CODEBOOK_PATH),
            "python_validator_sha256": sha256_path(PYTHON_VALIDATOR_PATH),
        },
        "items": private_items,
    }


def ensure_same_or_write(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"refusing to replace differing artifact: {path}")
        return
    atomic_text(path, text)


def connect_state() -> sqlite3.Connection:
    connection = sqlite3.connect(STATE_PATH)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS probes (
            judge_key TEXT PRIMARY KEY,
            verified_at_utc TEXT NOT NULL,
            reasoning_signal_present INTEGER NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            item_id TEXT NOT NULL,
            judge_key TEXT NOT NULL,
            route_role TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            start_timestamp_utc TEXT NOT NULL,
            end_timestamp_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            failure_kind TEXT,
            validation_error TEXT,
            raw_judge_response TEXT,
            response_sha256 TEXT,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (item_id, judge_key, route_role, attempt_number)
        );
        CREATE TABLE IF NOT EXISTS annotations (
            item_id TEXT PRIMARY KEY,
            judge_id TEXT NOT NULL,
            judge_family TEXT NOT NULL,
            judge_key TEXT NOT NULL,
            route_role TEXT NOT NULL,
            reused INTEGER NOT NULL,
            created_at_utc TEXT NOT NULL,
            annotation_json TEXT NOT NULL
        );
    """)
    return connection


def strict_annotation(
    *, item: dict[str, Any], semantic_source: dict[str, Any], coder_id: str,
    schema_validator: Draft202012Validator,
) -> dict[str, Any]:
    record = {
        "item_id": item["item_id"],
        "message": item["message"],
        "codebook_version": CODEBOOK_VERSION,
        **{field: semantic_source[field] for field in SEMANTIC_FIELDS},
        "coder_id": coder_id,
    }
    errors = sorted(schema_validator.iter_errors(record), key=lambda error: list(error.path))
    if errors:
        raise ValueError("JSON Schema: " + "; ".join(error.message for error in errors))
    validate_annotation_record(record)
    return record


def load_reused_annotations(
    items: list[dict[str, Any]], primary_key: dict[str, Any],
    schema_validator: Draft202012Validator,
) -> list[tuple[dict[str, Any], str, str]]:
    full_by_source = {
        (item["source_trial_id"], item["source_message_event_id"]): item for item in items
    }
    primary_by_id = {item["item_id"]: item for item in primary_key["items"]}
    reused: list[tuple[dict[str, Any], str, str]] = []
    seen: set[str] = set()
    for expected_judge_id, (path, expected_count, judge_family) in REUSED_FILES.items():
        rows = read_jsonl(path)
        if len(rows) != expected_count:
            raise ValueError(f"unexpected reused count in {path}")
        for row in rows:
            if row.get("judge_id") != expected_judge_id:
                raise ValueError(f"judge id mismatch in {path}")
            validation_item = primary_by_id[row["item_id"]]
            source_identity = (
                validation_item["source_trial_id"],
                validation_item["source_message_event_id"],
            )
            item = full_by_source[source_identity]
            if item["item_id"] in seen:
                raise ValueError("reused designated annotations overlap")
            if item["source_model_family"] == judge_family:
                raise ValueError("same-family reused annotation detected")
            annotation = strict_annotation(
                item=item, semantic_source=row, coder_id=expected_judge_id,
                schema_validator=schema_validator,
            )
            reused.append((annotation, expected_judge_id, judge_family))
            seen.add(item["item_id"])
    if len(reused) != EXPECTED_REUSED:
        raise ValueError(f"expected 130 reusable annotations, found {len(reused)}")
    return reused


def initial_manifest(items: list[dict[str, Any]], key: dict[str, Any]) -> dict[str, Any]:
    configurations = {}
    for name, spec in JUDGES.items():
        config = create_model_config(
            provider=spec.provider, model_alias=spec.model_alias,
            reasoning_mode=spec.reasoning_mode,
        )
        configurations[name] = {
            "judge_id": spec.judge_id,
            "judge_family": spec.family,
            "provider": config.provider,
            "model_alias": config.model_alias,
            "model_id": config.model_id,
            "reasoning_mode": config.reasoning_mode,
            "reasoning_request": config.provider_request_kwargs,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
    family_counts = Counter(item["source_model_family"] for item in items)
    return {
        "artifact_version": "promise_switch_full_semantic_coding_manifest_v1",
        "codebook_version": CODEBOOK_VERSION,
        "frozen_codebook_commit": FROZEN_CODEBOOK_COMMIT,
        "persisted_judge_runner_commit": PERSISTED_RUNNER_COMMIT,
        "start_timestamp_utc": utc_now(),
        "end_timestamp_utc": None,
        "eligible_population": len(items),
        "source_family_counts": dict(sorted(family_counts.items())),
        "reused_designated_validation_labels": EXPECTED_REUSED,
        "new_items_expected": len(items) - EXPECTED_REUSED,
        "prompt_sha256": key["file_hashes"]["judge_prompt_sha256"],
        "schema_sha256": key["file_hashes"]["annotation_schema_sha256"],
        "python_validator_sha256": key["file_hashes"]["python_validator_sha256"],
        "judge_configurations": configurations,
        "routing_rules": {
            "gpt_oss_source_primary": "deepseek_primary",
            "other_source_primary": "gpt_primary",
            "gpt_primary_non_deepseek_source_fallback": "deepseek_fallback",
            "gpt_primary_deepseek_source_fallback": "glm_fallback",
            "deepseek_primary_fallback": "glm_fallback",
            "same_family_judging_allowed": False,
        },
        "retry_policy": {
            "maximum_total_attempts_per_route": MAX_ATTEMPTS,
            "retryable": [
                "api_or_infrastructure_failure", "structural_or_schema_invalid_output",
            ],
            "semantic_surprise_retryable": False,
            "manual_repair_allowed": False,
        },
        "probes": {},
        "counts": {},
        "semantic_counts": {},
        "primary_promise_conditioned_subset": {},
        "source_family_by_judge_family": {},
    }


def prepare() -> None:
    items, primary_key = build_universe()
    csv_text = universe_csv(items)
    key = private_key(items, primary_key, csv_text)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if UNIVERSE_PATH.exists() or PRIVATE_KEY_PATH.exists():
        if not UNIVERSE_PATH.exists() or not PRIVATE_KEY_PATH.exists():
            raise ValueError("full-corpus public/private artifact pair is incomplete")
        saved_key = json.loads(PRIVATE_KEY_PATH.read_text(encoding="utf-8"))
        if sha256_path(UNIVERSE_PATH) != saved_key["file_hashes"]["universe_csv_sha256"]:
            raise ValueError("saved universe hash does not match its private key")
        if [item["item_id"] for item in saved_key["items"]] != [
            item["item_id"] for item in items
        ]:
            raise ValueError("saved private key differs from reconstructed universe")
        key = saved_key
    else:
        atomic_text(UNIVERSE_PATH, csv_text)
        atomic_json(PRIVATE_KEY_PATH, key)
    schema_validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    reused = load_reused_annotations(items, primary_key, schema_validator)
    connection = connect_state()
    with connection:
        if connection.execute(
            "SELECT 1 FROM attempts WHERE judge_key = 'qwen_fallback' LIMIT 1"
        ).fetchone():
            raise ValueError("cannot replace Qwen after a corpus attempt has used it")
        connection.execute("DELETE FROM probes WHERE judge_key = 'qwen_fallback'")
        for annotation, judge_id, judge_family in reused:
            existing = connection.execute(
                "SELECT annotation_json FROM annotations WHERE item_id = ?",
                (annotation["item_id"],),
            ).fetchone()
            serialized = json.dumps(annotation, ensure_ascii=False, sort_keys=True)
            if existing and existing[0] != serialized:
                raise ValueError("saved reused annotation differs from frozen input")
            connection.execute(
                """INSERT OR IGNORE INTO annotations
                   (item_id, judge_id, judge_family, judge_key, route_role, reused,
                    created_at_utc, annotation_json)
                   VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                (
                    annotation["item_id"], judge_id, judge_family, judge_id,
                    "reused_validation", utc_now(), serialized,
                ),
            )
    connection.close()
    refreshed = initial_manifest(items, key)
    if MANIFEST_PATH.exists():
        existing_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        refreshed["start_timestamp_utc"] = existing_manifest["start_timestamp_utc"]
        refreshed["probes"] = {
            name: value for name, value in existing_manifest.get("probes", {}).items()
            if name in {
                "gpt_primary", "deepseek_primary", "glm_fallback", "minimax_recovery",
            }
        }
        refreshed["counts"] = existing_manifest.get("counts", {})
        refreshed["semantic_counts"] = existing_manifest.get("semantic_counts", {})
        refreshed["primary_promise_conditioned_subset"] = existing_manifest.get(
            "primary_promise_conditioned_subset", {}
        )
        refreshed["source_family_by_judge_family"] = existing_manifest.get(
            "source_family_by_judge_family", {}
        )
    atomic_json(MANIFEST_PATH, refreshed)
    print(json.dumps({
        "eligible_population": len(items),
        "reused_designated_labels": len(reused),
        "new_items_requiring_judging": len(items) - len(reused),
        "universe_path": str(UNIVERSE_PATH),
        "private_key_path": str(PRIVATE_KEY_PATH),
    }, indent=2))


def load_items() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    key = json.loads(PRIVATE_KEY_PATH.read_text(encoding="utf-8"))
    with UNIVERSE_PATH.open(newline="", encoding="utf-8-sig") as handle:
        public_rows = list(csv.DictReader(handle))
    private_by_id = {item["item_id"]: item for item in key["items"]}
    if len(public_rows) != EXPECTED_ELIGIBLE or len(private_by_id) != EXPECTED_ELIGIBLE:
        raise ValueError("saved full corpus does not contain 2738 unique items")
    items = []
    for public in public_rows:
        private = private_by_id[public["item_id"]]
        if public["codebook_version"] != CODEBOOK_VERSION:
            raise ValueError("full corpus codebook version mismatch")
        if sha256_bytes(public["message"].encode("utf-8")) != private["message_sha256"]:
            raise ValueError("full corpus message hash mismatch")
        items.append({**private, "message": public["message"]})
    return items, {item["item_id"]: item for item in items}


def judge_client(spec: JudgeSpec, timeout_seconds: float) -> tuple[Any, ProviderAgnosticClient]:
    config = create_model_config(
        provider=spec.provider, model_alias=spec.model_alias,
        reasoning_mode=spec.reasoning_mode,
    )
    if config.reasoning_mode != "on":
        raise ValueError(f"judge {spec.key} is not reasoning-on")
    return config, ProviderAgnosticClient(
        config, OpenAICompatibleTransport(timeout_seconds=timeout_seconds),
    )


def expected_reasoning_field(spec: JudgeSpec) -> str:
    return "reasoning" if spec.family == "gpt_oss" else "reasoning_content"


def reasoning_present(spec: JudgeSpec, response: Any) -> bool:
    fields = response.error_metadata.get("reasoning_signal_fields", [])
    return expected_reasoning_field(spec) in fields


def run_probes(timeout_seconds: float) -> None:
    connection = connect_state()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for key in ("gpt_primary", "deepseek_primary", "glm_fallback"):
        spec = JUDGES[key]
        existing = connection.execute(
            "SELECT reasoning_signal_present, metadata_json FROM probes WHERE judge_key = ?",
            (key,),
        ).fetchone()
        if existing and probe_verified(connection, key):
            manifest["probes"][key] = json.loads(existing[1])
            continue
        config, client = judge_client(spec, timeout_seconds)
        started = utc_now()
        response = client.complete(
            messages=[{
                "role": "user",
                "content": "Non-scientific configuration probe. Reply with exactly OK.",
            }],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        present = response.status == "success" and reasoning_present(spec, response)
        metadata = {
            "verified_at_utc": utc_now(),
            "start_timestamp_utc": started,
            "status": response.status,
            "http_status": response.http_status,
            "provider_response_id": response.provider_response_id,
            "provider_request_id": response.provider_request_id,
            "finish_reason": response.finish_reason,
            "latency_seconds": response.latency_seconds,
            "usage": response.usage,
            "reasoning_signal_present": present,
            "expected_reasoning_signal_field": expected_reasoning_field(spec),
            "reasoning_signal_fields": response.error_metadata.get(
                "reasoning_signal_fields", []
            ),
            "reasoning_trace_present": bool(
                response.error_metadata.get("reasoning_trace_present")
            ),
            "provider": config.provider,
            "model_alias": config.model_alias,
            "model_id": config.model_id,
            "base_url": config.base_url,
            "reasoning_mode": config.reasoning_mode,
            "reasoning_request": config.provider_request_kwargs,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        with connection:
            connection.execute(
                """INSERT OR REPLACE INTO probes
                   (judge_key, verified_at_utc, reasoning_signal_present, metadata_json)
                   VALUES (?, ?, ?, ?)""",
                (key, metadata["verified_at_utc"], int(present), json.dumps(metadata)),
            )
        manifest["probes"][key] = metadata
        atomic_json(MANIFEST_PATH, manifest)
        if not present:
            connection.close()
            raise RuntimeError(f"reasoning-on probe failed for {key}: {response.status}")
    connection.close()
    atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest["probes"], indent=2, sort_keys=True))


def run_minimax_probe(timeout_seconds: float) -> None:
    key = "minimax_recovery"
    spec = JUDGES[key]
    connection = connect_state()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing = connection.execute(
        "SELECT metadata_json FROM probes WHERE judge_key = ?", (key,)
    ).fetchone()
    if existing and probe_verified(connection, key):
        metadata = json.loads(existing[0])
    else:
        config, client = judge_client(spec, timeout_seconds)
        started = utc_now()
        response = client.complete(
            messages=[{
                "role": "user",
                "content": "Non-scientific configuration probe. Reply with exactly OK.",
            }],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        present = response.status == "success" and reasoning_present(spec, response)
        metadata = {
            "verified_at_utc": utc_now(),
            "start_timestamp_utc": started,
            "status": response.status,
            "http_status": response.http_status,
            "provider_response_id": response.provider_response_id,
            "provider_request_id": response.provider_request_id,
            "finish_reason": response.finish_reason,
            "latency_seconds": response.latency_seconds,
            "usage": response.usage,
            "reasoning_signal_present": present,
            "expected_reasoning_signal_field": expected_reasoning_field(spec),
            "reasoning_signal_fields": response.error_metadata.get(
                "reasoning_signal_fields", []
            ),
            "reasoning_trace_present": bool(
                response.error_metadata.get("reasoning_trace_present")
            ),
            "provider": config.provider,
            "model_alias": config.model_alias,
            "model_id": config.model_id,
            "base_url": config.base_url,
            "reasoning_mode": config.reasoning_mode,
            "reasoning_request": config.provider_request_kwargs,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        with connection:
            connection.execute(
                """INSERT OR REPLACE INTO probes
                   (judge_key, verified_at_utc, reasoning_signal_present, metadata_json)
                   VALUES (?, ?, ?, ?)""",
                (key, metadata["verified_at_utc"], int(present), json.dumps(metadata)),
            )
    connection.close()
    manifest.setdefault("probes", {})[key] = metadata
    config = create_model_config(
        provider=spec.provider, model_alias=spec.model_alias,
        reasoning_mode=spec.reasoning_mode,
    )
    manifest.setdefault("judge_configurations", {})[key] = {
        "judge_id": spec.judge_id,
        "judge_family": spec.family,
        "provider": config.provider,
        "model_alias": config.model_alias,
        "model_id": config.model_id,
        "reasoning_mode": config.reasoning_mode,
        "reasoning_request": config.provider_request_kwargs,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    atomic_json(MANIFEST_PATH, manifest)
    if not metadata["reasoning_signal_present"]:
        raise RuntimeError(f"reasoning-on probe failed for {key}: {metadata['status']}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def primary_route(item: dict[str, Any]) -> tuple[str, str]:
    if item["source_model_family"] == "gpt_oss":
        return "deepseek_primary", "primary"
    return "gpt_primary", "primary"


def fallback_route(item: dict[str, Any], primary_key: str) -> tuple[str, str]:
    if primary_key == "deepseek_primary":
        return "glm_fallback", "fallback"
    if item["source_model_family"] == "deepseek":
        return "glm_fallback", "fallback"
    return "deepseek_fallback", "fallback"


def attempt_counts(connection: sqlite3.Connection, item_id: str) -> dict[tuple[str, str], int]:
    rows = connection.execute(
        """SELECT judge_key, route_role, COUNT(*) FROM attempts
           WHERE item_id = ? GROUP BY judge_key, route_role""",
        (item_id,),
    ).fetchall()
    return {(row[0], row[1]): int(row[2]) for row in rows}


def next_job(
    connection: sqlite3.Connection, item: dict[str, Any], *, delay: float = 0.0,
) -> dict[str, Any] | None:
    if connection.execute(
        "SELECT 1 FROM annotations WHERE item_id = ?", (item["item_id"],)
    ).fetchone():
        return None
    counts = attempt_counts(connection, item["item_id"])
    primary_key, primary_role = primary_route(item)
    primary_count = counts.get((primary_key, primary_role), 0)
    if primary_count < MAX_ATTEMPTS:
        judge_key, route_role, number = primary_key, primary_role, primary_count + 1
    else:
        judge_key, route_role = fallback_route(item, primary_key)
        fallback_count = counts.get((judge_key, route_role), 0)
        if fallback_count >= MAX_ATTEMPTS:
            return None
        number = fallback_count + 1
    spec = JUDGES[judge_key]
    if spec.family == item["source_model_family"]:
        raise ValueError(f"same-family route for {item['item_id']}")
    if not probe_verified(connection, probe_key(judge_key)):
        if route_role == "primary":
            raise RuntimeError(f"verified reasoning-on probe missing for {judge_key}")
        return None
    return {
        "item_id": item["item_id"],
        "judge_key": judge_key,
        "route_role": route_role,
        "attempt_number": number,
        "delay": delay,
    }


def call_once(
    *, job: dict[str, Any], item: dict[str, Any], client: ProviderAgnosticClient,
    prompt_template: str,
) -> tuple[dict[str, Any], Any, str, str]:
    if job["delay"]:
        time.sleep(job["delay"])
    prompt = prompt_template.replace("{{ITEM_ID}}", item["item_id"]).replace(
        "{{MESSAGE}}", item["message"]
    )
    started = utc_now()
    response = client.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return job, response, started, utc_now()


def classify_response(
    *, response: Any, item: dict[str, Any], spec: JudgeSpec,
    schema_validator: Draft202012Validator,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if response.status != "success":
        return None, "infrastructure", "API/infrastructure failure"
    if not reasoning_present(spec, response):
        return None, "infrastructure", "successful response lacked reasoning signal"
    try:
        parsed = json.loads(response.raw_response)
        if not isinstance(parsed, dict):
            raise ValueError("judge output must be one JSON object")
        errors = sorted(schema_validator.iter_errors(parsed), key=lambda error: list(error.path))
        if errors:
            raise ValueError("JSON Schema: " + "; ".join(error.message for error in errors))
        validate_annotation_record(parsed)
        if parsed["item_id"] != item["item_id"]:
            raise ValueError("item_id mismatch")
        if parsed["message"] != item["message"]:
            raise ValueError("message mismatch")
        annotation = strict_annotation(
            item=item, semantic_source=parsed, coder_id=spec.judge_id,
            schema_validator=schema_validator,
        )
        return annotation, None, None
    except Exception as exc:
        return None, "schema_or_structural", str(exc)


def classify_format_recovery_response(
    *, response: Any, item: dict[str, Any], spec: JudgeSpec,
    schema_validator: Draft202012Validator,
) -> tuple[dict[str, Any] | None, str | None, str | None, dict[str, bool | None]]:
    stages: dict[str, bool | None] = {
        "raw_json_parse_valid": None,
        "schema_valid": None,
        "python_validator_valid": None,
    }
    if response.status != "success":
        return None, "infrastructure", "API/infrastructure failure", stages
    if not reasoning_present(spec, response):
        return None, "infrastructure", "successful response lacked reasoning signal", stages
    try:
        parsed = json.loads(response.raw_response)
        stages["raw_json_parse_valid"] = True
    except Exception as exc:
        stages["raw_json_parse_valid"] = False
        return None, "schema_or_structural", str(exc), stages
    if not isinstance(parsed, dict):
        return None, "schema_or_structural", "judge output must be one JSON object", stages
    errors = sorted(schema_validator.iter_errors(parsed), key=lambda error: list(error.path))
    if errors:
        stages["schema_valid"] = False
        return (
            None,
            "schema_or_structural",
            "JSON Schema: " + "; ".join(error.message for error in errors),
            stages,
        )
    stages["schema_valid"] = True
    try:
        validate_annotation_record(parsed)
    except Exception as exc:
        stages["python_validator_valid"] = False
        return None, "schema_or_structural", str(exc), stages
    stages["python_validator_valid"] = True
    if parsed["item_id"] != item["item_id"]:
        return None, "schema_or_structural", "item_id mismatch", stages
    if parsed["message"] != item["message"]:
        return None, "schema_or_structural", "message mismatch", stages
    annotation = strict_annotation(
        item=item, semantic_source=parsed, coder_id=spec.judge_id,
        schema_validator=schema_validator,
    )
    return annotation, None, None, stages


def record_result(
    connection: sqlite3.Connection, *, job: dict[str, Any], response: Any,
    started: str, ended: str, annotation: dict[str, Any] | None,
    failure_kind: str | None, validation_error: str | None,
) -> None:
    spec = JUDGES[job["judge_key"]]
    config = create_model_config(
        provider=spec.provider, model_alias=spec.model_alias,
        reasoning_mode=spec.reasoning_mode,
    )
    metadata = {
        "http_status": response.http_status,
        "provider_response_id": response.provider_response_id,
        "provider_request_id": response.provider_request_id,
        "finish_reason": response.finish_reason,
        "latency_seconds": response.latency_seconds,
        "retry_after_seconds": response.retry_after_seconds,
        "usage": response.usage,
        "reasoning_signal_present": reasoning_present(spec, response),
        "reasoning_signal_fields": response.error_metadata.get(
            "reasoning_signal_fields", []
        ),
        "reasoning_trace_present": bool(response.error_metadata.get("reasoning_trace_present")),
        "error_metadata": response.error_metadata if response.status != "success" else {},
        "provider": config.provider,
        "model_alias": config.model_alias,
        "model_id": config.model_id,
        "base_url": config.base_url,
        "reasoning_mode": config.reasoning_mode,
        "reasoning_request": config.provider_request_kwargs,
        "expected_reasoning_signal_field": expected_reasoning_field(spec),
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if "validation_stages" in job:
        metadata["validation_stages"] = job["validation_stages"]
    raw_failure = response.raw_response if annotation is None else None
    response_sha = (
        sha256_bytes(response.raw_response.encode("utf-8"))
        if response.raw_response else None
    )
    with connection:
        connection.execute(
            """INSERT INTO attempts
               (item_id, judge_key, route_role, attempt_number, start_timestamp_utc,
                end_timestamp_utc, status, failure_kind, validation_error,
                raw_judge_response, response_sha256, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job["item_id"], job["judge_key"], job["route_role"],
                job["attempt_number"], started, ended,
                "valid" if annotation is not None else "invalid",
                failure_kind, validation_error, raw_failure, response_sha,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        if annotation is not None:
            connection.execute(
                """INSERT INTO annotations
                   (item_id, judge_id, judge_family, judge_key, route_role, reused,
                    created_at_utc, annotation_json)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?)""",
                (
                    job["item_id"], spec.judge_id, spec.family, spec.key,
                    job["route_role"], ended,
                    json.dumps(annotation, ensure_ascii=False, sort_keys=True),
                ),
            )


def probe_verified(connection: sqlite3.Connection, judge_key: str) -> bool:
    row = connection.execute(
        "SELECT reasoning_signal_present, metadata_json FROM probes WHERE judge_key = ?",
        (judge_key,),
    ).fetchone()
    if not row or row[0] != 1:
        return False
    spec = JUDGES[judge_key]
    config = create_model_config(
        provider=spec.provider, model_alias=spec.model_alias,
        reasoning_mode=spec.reasoning_mode,
    )
    metadata = json.loads(row[1])
    expected_field = expected_reasoning_field(spec)
    return (
        metadata.get("status") == "success"
        and metadata.get("http_status") == 200
        and metadata.get("provider") == config.provider
        and metadata.get("model_id") == config.model_id
        and metadata.get("reasoning_mode") == "on"
        and metadata.get("reasoning_request") == config.provider_request_kwargs
        and metadata.get("temperature") == TEMPERATURE
        and metadata.get("max_output_tokens") == MAX_OUTPUT_TOKENS
        and metadata.get("reasoning_signal_present") is True
        and expected_field in metadata.get("reasoning_signal_fields", [])
    )


def probe_key(judge_key: str) -> str:
    if judge_key == "deepseek_fallback":
        return "deepseek_primary"
    return judge_key


def require_primary_probes(connection: sqlite3.Connection) -> None:
    for key in ("gpt_primary", "deepseek_primary"):
        if not probe_verified(connection, key):
            raise RuntimeError(f"verified reasoning-on probe missing for {key}")


def execute(*, workers: int, timeout_seconds: float, progress_every: int) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    items, item_by_id = load_items()
    connection = connect_state()
    require_primary_probes(connection)
    reused_ids = {
        row[0] for row in connection.execute(
            "SELECT item_id FROM annotations WHERE reused = 1"
        ).fetchall()
    }
    if len(reused_ids) != EXPECTED_REUSED:
        raise ValueError("state does not contain 130 reused annotations")
    queue: deque[dict[str, Any]] = deque()
    for item in items:
        if item["item_id"] in reused_ids:
            continue
        job = next_job(connection, item)
        if job is not None:
            queue.append(job)
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    schema_validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    clients = {
        key: judge_client(spec, timeout_seconds)[1] for key, spec in JUDGES.items()
    }
    completed_at_start = connection.execute(
        "SELECT COUNT(*) FROM annotations WHERE reused = 0"
    ).fetchone()[0]
    completed_this_run = 0
    futures: dict[Future[Any], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while queue or futures:
            while queue and len(futures) < workers:
                job = queue.popleft()
                if not probe_verified(connection, probe_key(job["judge_key"])):
                    raise RuntimeError(
                        f"verified reasoning-on probe missing for {job['judge_key']}"
                    )
                item = item_by_id[job["item_id"]]
                future = executor.submit(
                    call_once, job=job, item=item, client=clients[job["judge_key"]],
                    prompt_template=prompt_template,
                )
                futures[future] = job
            if not futures:
                break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                job, response, started, ended = future.result()
                item = item_by_id[job["item_id"]]
                annotation, failure_kind, validation_error = classify_response(
                    response=response, item=item, spec=JUDGES[job["judge_key"]],
                    schema_validator=schema_validator,
                )
                record_result(
                    connection, job=job, response=response, started=started, ended=ended,
                    annotation=annotation, failure_kind=failure_kind,
                    validation_error=validation_error,
                )
                if annotation is not None:
                    completed_this_run += 1
                    total = completed_at_start + completed_this_run
                    if completed_this_run % progress_every == 0:
                        print(f"new valid annotations {total}/2608", flush=True)
                else:
                    retry_after = response.retry_after_seconds or 0.0
                    delay = min(max(float(retry_after), 2 ** job["attempt_number"]), 60.0)
                    follow_up = next_job(connection, item, delay=delay)
                    if follow_up is not None:
                        queue.append(follow_up)
    connection.close()
    finalize()


def export_minimax_attempts(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """SELECT item_id, attempt_number, start_timestamp_utc, end_timestamp_utc,
                  status, failure_kind, validation_error, response_sha256, metadata_json
           FROM attempts WHERE judge_key = 'minimax_recovery'
           ORDER BY item_id, attempt_number"""
    ).fetchall()
    resolved = {
        row[0] for row in connection.execute(
            "SELECT item_id FROM annotations WHERE judge_key = 'minimax_recovery'"
        ).fetchall()
    }
    spec = JUDGES["minimax_recovery"]
    records = []
    for row in rows:
        metadata = json.loads(row[8])
        records.append({
            "item_id": row[0],
            "assigned_judge": spec.judge_id,
            "judge_family": spec.family,
            "attempt_number": row[1],
            "start_timestamp_utc": row[2],
            "end_timestamp_utc": row[3],
            "attempt_status": row[4],
            "failure_kind": row[5],
            "validation_error": row[6],
            "response_sha256": row[7],
            "provider": metadata["provider"],
            "model_alias": metadata["model_alias"],
            "model_id": metadata["model_id"],
            "base_url": metadata["base_url"],
            "reasoning_mode": metadata["reasoning_mode"],
            "reasoning_request": metadata["reasoning_request"],
            "temperature": metadata["temperature"],
            "max_output_tokens": metadata["max_output_tokens"],
            "http_status": metadata["http_status"],
            "provider_response_id": metadata["provider_response_id"],
            "provider_request_id": metadata["provider_request_id"],
            "reasoning_signal_present": metadata["reasoning_signal_present"],
            "reasoning_signal_fields": metadata["reasoning_signal_fields"],
            "schema_valid": row[4] == "valid",
            "python_valid": row[4] == "valid",
            "item_final_status": "resolved" if row[0] in resolved else "unresolved",
        })
    atomic_jsonl(MINIMAX_ATTEMPTS_PATH, records)


def recover_with_minimax(timeout_seconds: float) -> None:
    items, item_by_id = load_items()
    connection = connect_state()
    unresolved = {
        item["item_id"] for item in items
        if not connection.execute(
            "SELECT 1 FROM annotations WHERE item_id = ?", (item["item_id"],)
        ).fetchone()
    }
    targets = set(MINIMAX_RECOVERY_IDS)
    if not unresolved.issubset(targets):
        connection.close()
        raise ValueError("unresolved corpus contains items outside the Judge-4 target set")
    for item_id in MINIMAX_RECOVERY_IDS:
        item = item_by_id[item_id]
        if item["source_model_family"] == JUDGES["minimax_recovery"].family:
            connection.close()
            raise ValueError(f"same-family MiniMax recovery route for {item_id}")
        prior = connection.execute(
            """SELECT judge_key, COUNT(*) FROM attempts WHERE item_id = ?
               AND judge_key IN ('gpt_primary', 'glm_fallback') GROUP BY judge_key""",
            (item_id,),
        ).fetchall()
        if item_id in unresolved and dict(prior) != {"gpt_primary": 3, "glm_fallback": 3}:
            connection.close()
            raise ValueError(f"prior attempt history is incomplete for {item_id}")
    if not probe_verified(connection, "minimax_recovery"):
        connection.close()
        raise RuntimeError("verified reasoning-on probe missing for minimax_recovery")
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    schema_validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    _, client = judge_client(JUDGES["minimax_recovery"], timeout_seconds)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recovery = manifest.setdefault("judge4_recovery", {})
    recovery.setdefault("start_timestamp_utc", utc_now())
    recovery["target_item_ids"] = list(MINIMAX_RECOVERY_IDS)
    recovery["source_family_counts"] = {"deepseek": len(MINIMAX_RECOVERY_IDS)}
    recovery["attempt_provenance_path"] = str(MINIMAX_ATTEMPTS_PATH)
    atomic_json(MANIFEST_PATH, manifest)
    for item_id in MINIMAX_RECOVERY_IDS:
        if connection.execute(
            "SELECT 1 FROM annotations WHERE item_id = ?", (item_id,)
        ).fetchone():
            continue
        attempt_number = connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE item_id = ? AND judge_key = ?",
            (item_id, "minimax_recovery"),
        ).fetchone()[0]
        while attempt_number < MAX_ATTEMPTS:
            attempt_number += 1
            job = {
                "item_id": item_id,
                "judge_key": "minimax_recovery",
                "route_role": "recovery_judge4",
                "attempt_number": attempt_number,
                "delay": 0.0 if attempt_number == 1 else min(2 ** (attempt_number - 1), 60),
            }
            _, response, started, ended = call_once(
                job=job, item=item_by_id[item_id], client=client,
                prompt_template=prompt_template,
            )
            annotation, failure_kind, validation_error = classify_response(
                response=response, item=item_by_id[item_id],
                spec=JUDGES["minimax_recovery"], schema_validator=schema_validator,
            )
            record_result(
                connection, job=job, response=response, started=started, ended=ended,
                annotation=annotation, failure_kind=failure_kind,
                validation_error=validation_error,
            )
            export_minimax_attempts(connection)
            if annotation is not None:
                break
    export_minimax_attempts(connection)
    remaining = [
        item_id for item_id in MINIMAX_RECOVERY_IDS
        if not connection.execute(
            "SELECT 1 FROM annotations WHERE item_id = ?", (item_id,)
        ).fetchone()
    ]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["judge4_recovery"].update({
        "end_timestamp_utc": utc_now(),
        "resolved": len(MINIMAX_RECOVERY_IDS) - len(remaining),
        "unresolved": len(remaining),
        "unresolved_item_ids": remaining,
    })
    atomic_json(MANIFEST_PATH, manifest)
    connection.close()
    finalize()


def export_format_recovery_attempts(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """SELECT item_id, attempt_number, start_timestamp_utc, end_timestamp_utc,
                  status, failure_kind, validation_error, response_sha256, metadata_json
           FROM attempts WHERE judge_key = 'minimax_recovery'
             AND route_role = 'format_only_recovery'
           ORDER BY item_id, attempt_number"""
    ).fetchall()
    resolved = {
        row[0] for row in connection.execute(
            "SELECT item_id FROM annotations WHERE item_id IN (?, ?)", FORMAT_RECOVERY_IDS
        ).fetchall()
    }
    spec = JUDGES["minimax_recovery"]
    records = []
    for row in rows:
        metadata = json.loads(row[8])
        stages = metadata["validation_stages"]
        records.append({
            "recovery_type": "format_only_recovery",
            "item_id": row[0],
            "assigned_judge": spec.judge_id,
            "judge_family": spec.family,
            "attempt_number": row[1],
            "start_timestamp_utc": row[2],
            "end_timestamp_utc": row[3],
            "attempt_status": row[4],
            "failure_kind": row[5],
            "validation_error": row[6],
            "response_sha256": row[7],
            "provider": metadata["provider"],
            "model_alias": metadata["model_alias"],
            "model_id": metadata["model_id"],
            "base_url": metadata["base_url"],
            "reasoning_mode": metadata["reasoning_mode"],
            "reasoning_request": metadata["reasoning_request"],
            "temperature": metadata["temperature"],
            "max_output_tokens": metadata["max_output_tokens"],
            "http_status": metadata["http_status"],
            "provider_response_id": metadata["provider_response_id"],
            "provider_request_id": metadata["provider_request_id"],
            "reasoning_signal_present": metadata["reasoning_signal_present"],
            "reasoning_signal_fields": metadata["reasoning_signal_fields"],
            **stages,
            "item_final_status": "resolved" if row[0] in resolved else "unresolved",
        })
    atomic_jsonl(FORMAT_RECOVERY_ATTEMPTS_PATH, records)


def recover_format_only(timeout_seconds: float) -> None:
    items, item_by_id = load_items()
    connection = connect_state()
    unresolved = {
        item["item_id"] for item in items
        if not connection.execute(
            "SELECT 1 FROM annotations WHERE item_id = ?", (item["item_id"],)
        ).fetchone()
    }
    targets = set(FORMAT_RECOVERY_IDS)
    if not unresolved.issubset(targets):
        connection.close()
        raise ValueError("unresolved corpus contains items outside the format-recovery set")
    for item_id in FORMAT_RECOVERY_IDS:
        item = item_by_id[item_id]
        if item["source_model_family"] == JUDGES["minimax_recovery"].family:
            connection.close()
            raise ValueError(f"same-family MiniMax format recovery for {item_id}")
        prior = connection.execute(
            """SELECT COUNT(*) FROM attempts WHERE item_id = ?
               AND judge_key = 'minimax_recovery' AND route_role = 'recovery_judge4'""",
            (item_id,),
        ).fetchone()[0]
        if item_id in unresolved and prior != MAX_ATTEMPTS:
            connection.close()
            raise ValueError(f"Judge-4 attempt history is incomplete for {item_id}")
    if not probe_verified(connection, "minimax_recovery"):
        connection.close()
        raise RuntimeError("verified reasoning-on probe missing for minimax_recovery")
    prompt_template = (
        PROMPT_PATH.read_text(encoding="utf-8") + "\n\n" + FORMAT_ONLY_INSTRUCTION
    )
    schema_validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    _, client = judge_client(JUDGES["minimax_recovery"], timeout_seconds)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recovery = manifest.setdefault("format_only_recovery", {})
    recovery.setdefault("start_timestamp_utc", utc_now())
    recovery["target_item_ids"] = list(FORMAT_RECOVERY_IDS)
    recovery["prompt_addition"] = FORMAT_ONLY_INSTRUCTION
    recovery["attempt_provenance_path"] = str(FORMAT_RECOVERY_ATTEMPTS_PATH)
    atomic_json(MANIFEST_PATH, manifest)
    for item_id in FORMAT_RECOVERY_IDS:
        if connection.execute(
            "SELECT 1 FROM annotations WHERE item_id = ?", (item_id,)
        ).fetchone():
            continue
        attempt_number = connection.execute(
            """SELECT COUNT(*) FROM attempts WHERE item_id = ? AND judge_key = ?
               AND route_role = 'format_only_recovery'""",
            (item_id, "minimax_recovery"),
        ).fetchone()[0]
        while attempt_number < MAX_ATTEMPTS:
            attempt_number += 1
            job = {
                "item_id": item_id,
                "judge_key": "minimax_recovery",
                "route_role": "format_only_recovery",
                "attempt_number": attempt_number,
                "delay": 0.0 if attempt_number == 1 else min(2 ** (attempt_number - 1), 60),
            }
            _, response, started, ended = call_once(
                job=job, item=item_by_id[item_id], client=client,
                prompt_template=prompt_template,
            )
            annotation, failure_kind, validation_error, stages = (
                classify_format_recovery_response(
                    response=response, item=item_by_id[item_id],
                    spec=JUDGES["minimax_recovery"], schema_validator=schema_validator,
                )
            )
            job["validation_stages"] = stages
            record_result(
                connection, job=job, response=response, started=started, ended=ended,
                annotation=annotation, failure_kind=failure_kind,
                validation_error=validation_error,
            )
            export_format_recovery_attempts(connection)
            if annotation is not None:
                break
    export_format_recovery_attempts(connection)
    remaining = [
        item_id for item_id in FORMAT_RECOVERY_IDS
        if not connection.execute(
            "SELECT 1 FROM annotations WHERE item_id = ?", (item_id,)
        ).fetchone()
    ]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["format_only_recovery"].update({
        "end_timestamp_utc": utc_now(),
        "resolved": len(FORMAT_RECOVERY_IDS) - len(remaining),
        "unresolved": len(remaining),
        "unresolved_item_ids": remaining,
    })
    atomic_json(MANIFEST_PATH, manifest)
    connection.close()
    finalize()


def validate_saved_annotation(
    record: dict[str, Any], item: dict[str, Any], schema_validator: Draft202012Validator,
) -> None:
    errors = list(schema_validator.iter_errors(record))
    if errors:
        raise ValueError("saved annotation is not JSON-Schema valid")
    validate_annotation_record(record)
    if record["item_id"] != item["item_id"] or record["message"] != item["message"]:
        raise ValueError("saved annotation does not match its blinded item")


def finalize() -> None:
    items, item_by_id = load_items()
    connection = connect_state()
    annotation_rows = connection.execute(
        """SELECT item_id, judge_id, judge_family, judge_key, route_role, reused,
                  created_at_utc, annotation_json FROM annotations"""
    ).fetchall()
    annotations = {row[0]: json.loads(row[7]) for row in annotation_rows}
    schema_validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    for item_id, annotation in annotations.items():
        validate_saved_annotation(annotation, item_by_id[item_id], schema_validator)
    attempts_db = connection.execute(
        """SELECT item_id, judge_key, route_role, attempt_number,
                  start_timestamp_utc, end_timestamp_utc, status, failure_kind,
                  validation_error, raw_judge_response, response_sha256, metadata_json
           FROM attempts ORDER BY start_timestamp_utc, item_id"""
    ).fetchall()
    attempts = [{
        "item_id": row[0], "judge_key": row[1], "route_role": row[2],
        "attempt_number": row[3], "start_timestamp_utc": row[4],
        "end_timestamp_utc": row[5], "status": row[6], "failure_kind": row[7],
        "validation_error": row[8], "raw_judge_response": row[9],
        "response_sha256": row[10], "metadata": json.loads(row[11]),
    } for row in attempts_db]
    atomic_jsonl(ATTEMPTS_PATH, attempts)
    by_judge: dict[str, list[dict[str, Any]]] = {}
    designations = []
    routing_table: dict[str, Counter[str]] = {}
    route_counts: Counter[str] = Counter()
    for row in annotation_rows:
        item_id, judge_id, judge_family, judge_key, route_role, reused, created_at, _ = row
        by_judge.setdefault(judge_id, []).append(annotations[item_id])
        item = item_by_id[item_id]
        if item["source_model_family"] == judge_family:
            raise ValueError(f"same-family final annotation for {item_id}")
        routing_table.setdefault(item["source_model_family"], Counter())[judge_family] += 1
        route_counts["reused" if reused else judge_key] += 1
        designations.append({
            "item_id": item_id,
            "source_trial_id": item["source_trial_id"],
            "source_message_event_id": item["source_message_event_id"],
            "source_model_family": item["source_model_family"],
            "judge_id": judge_id,
            "judge_family": judge_family,
            "judge_key": judge_key,
            "route_role": route_role,
            "reused_validation_annotation": bool(reused),
            "created_at_utc": created_at,
        })
    order = {item["item_id"]: index for index, item in enumerate(items)}
    for judge_id, rows in by_judge.items():
        rows.sort(key=lambda row: order[row["item_id"]])
        safe_name = re.sub(r"[^a-z0-9_]+", "_", judge_id.lower()).strip("_")
        atomic_jsonl(OUTPUT_DIR / f"{safe_name}_annotations.jsonl", rows)
    designations.sort(key=lambda row: order[row["item_id"]])
    atomic_jsonl(DESIGNATIONS_PATH, designations)
    unresolved = [item["item_id"] for item in items if item["item_id"] not in annotations]
    if not unresolved:
        atomic_jsonl(FINAL_PATH, [annotations[item["item_id"]] for item in items])
    elif FINAL_PATH.exists():
        raise ValueError("final consolidated artifact exists while unresolved items remain")
    semantic_counts = {}
    subset_summary = {}
    if not unresolved:
        for field in SEMANTIC_FIELDS:
            if field == "condition_text":
                continue
            semantic_counts[field] = dict(sorted(Counter(
                annotation[field] for annotation in annotations.values()
            ).items()))
        subset_count = sum(
            eligible_explicit_promiser(MessageCode.from_mapping(annotation))
            for annotation in annotations.values()
        )
        subset_summary = {
            "count": subset_count,
            "percentage_of_eligible": 100.0 * subset_count / len(items),
            "definition": (
                "explicit_promise=yes AND promised_action=COOPERATE AND "
                "conditionality=unconditional AND ambiguous_response=no"
            ),
        }
    failure_counts = Counter(
        attempt["failure_kind"] for attempt in attempts if attempt["failure_kind"]
    )
    retry_counts = Counter()
    for attempt in attempts:
        if attempt["attempt_number"] > 1:
            retry_counts[attempt["route_role"]] += 1
    fallback_attempts = sum(attempt["route_role"] == "fallback" for attempt in attempts)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["end_timestamp_utc"] = utc_now() if not unresolved else None
    manifest["counts"] = {
        "eligible_population": len(items),
        "reused_designated_validation_labels": sum(row[5] for row in annotation_rows),
        "new_valid_annotations": sum(not row[5] for row in annotation_rows),
        "resolved": len(annotations),
        "unresolved": len(unresolved),
        "unresolved_item_ids": unresolved,
        "valid_annotations_by_route": dict(sorted(route_counts.items())),
        "total_api_attempts": len(attempts),
        "primary_retry_attempts": retry_counts["primary"],
        "fallback_retry_attempts": retry_counts["fallback"],
        "fallback_attempts": fallback_attempts,
        "schema_or_structural_invalid_attempts": failure_counts["schema_or_structural"],
        "infrastructure_failed_attempts": failure_counts["infrastructure"],
    }
    manifest["semantic_counts"] = semantic_counts
    manifest["primary_promise_conditioned_subset"] = subset_summary
    all_families = sorted({item["source_model_family"] for item in items})
    judge_families = sorted({row[2] for row in annotation_rows})
    manifest["source_family_by_judge_family"] = {
        source: {judge: routing_table.get(source, Counter())[judge] for judge in judge_families}
        for source in all_families
    }
    manifest["same_family_judging_count"] = sum(
        routing_table.get(family, Counter())[family] for family in all_families
    )
    atomic_json(MANIFEST_PATH, manifest)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    print(json.dumps({
        "resolved": len(annotations),
        "unresolved": len(unresolved),
        "new_valid_annotations": manifest["counts"]["new_valid_annotations"],
        "same_family_judging_count": manifest["same_family_judging_count"],
        "final_path": str(FINAL_PATH) if not unresolved else None,
    }, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action in {"prepare", "all"}:
        prepare()
    if args.action in {"probe", "all"}:
        run_probes(args.timeout_seconds)
    if args.action == "probe-minimax":
        run_minimax_probe(args.timeout_seconds)
    if args.action in {"run", "all"}:
        execute(
            workers=args.workers, timeout_seconds=args.timeout_seconds,
            progress_every=args.progress_every,
        )
    if args.action == "recover-minimax":
        recover_with_minimax(args.timeout_seconds)
    if args.action == "recover-format-only":
        recover_format_only(args.timeout_seconds)
    if args.action == "finalize":
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
