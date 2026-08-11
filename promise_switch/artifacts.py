"""Read-only validation for Promise Switch execution artifact directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .design import Condition, scientific_condition_label, treatment_metadata
from .schemas import ModelConfig
from .version import EXPERIMENT_VERSION, PROMPT_VERSION, prompt_templates_hash

EXPECTED_EXECUTION_SCHEMA = "promise_switch_execution_v2"
EXPECTED_EXPERIMENT_VERSION = "promise_switch_v1"
LEGACY_EXECUTION_SCHEMA = "promise_switch_execution_v1"
DATASET_ROLES = {"technical", "pilot", "confirmatory", "synthetic"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_run_directory(
    run_dir: Path,
    *,
    allow_invalid: bool = False,
    allow_legacy: bool = False,
    required_dataset_role: str | None = None,
) -> dict[str, Any]:
    """Validate immutable identity before a run can be audited or analyzed."""
    manifest_path = run_dir / "manifest.json"
    trials_path = run_dir / "trials.jsonl"
    if not manifest_path.is_file() or not trials_path.is_file():
        raise ValueError(f"not a complete Promise Switch execution run: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("execution_schema_version")
    if schema != EXPECTED_EXECUTION_SCHEMA:
        if not (allow_legacy and schema == LEGACY_EXECUTION_SCHEMA):
            raise ValueError(f"legacy or incompatible execution schema: {run_dir}")
        if (run_dir / "INVALID_CONCURRENT_EXECUTION.json").exists() and not allow_invalid:
            raise ValueError(f"run is marked INVALID_CONCURRENT_EXECUTION: {run_dir}")
        return manifest
    if manifest.get("experiment_version") != EXPECTED_EXPERIMENT_VERSION:
        raise ValueError(f"legacy or incompatible experiment version: {run_dir}")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"prompt version mismatch: {run_dir}")
    if manifest.get("prompt_hash") != prompt_templates_hash():
        raise ValueError(f"prompt hash mismatch: {run_dir}")
    role = manifest.get("dataset_role")
    if role not in DATASET_ROLES:
        raise ValueError(f"missing or invalid dataset role: {run_dir}")
    if required_dataset_role is not None and role != required_dataset_role:
        raise ValueError(
            f"dataset role {role!r} does not match required role "
            f"{required_dataset_role!r}: {run_dir}"
        )
    if bool(manifest.get("dry_run")) != (role == "synthetic"):
        raise ValueError(f"synthetic/dry-run classification mismatch: {run_dir}")
    if (run_dir / "INVALID_CONCURRENT_EXECUTION.json").exists() and not allow_invalid:
        raise ValueError(f"run is marked INVALID_CONCURRENT_EXECUTION: {run_dir}")
    return manifest


def validate_run_integrity(
    run_dir: Path,
    *,
    required_dataset_role: str | None = None,
    require_completed: bool = False,
) -> dict[str, Any]:
    """Reject duplicate, imbalanced, cross-linked, or corrupted current-schema runs."""
    from .execution.planner import PlannedTrial, plan_trials

    manifest = validate_run_directory(
        run_dir, required_dataset_role=required_dataset_role,
    )
    normalized = manifest.get("normalized_execution_parameters")
    if not isinstance(normalized, dict):
        raise ValueError(f"missing normalized execution parameters: {run_dir}")
    try:
        configs = tuple(ModelConfig(**row) for row in normalized["model_configurations"])
        n_per_condition = int(normalized["n_per_condition"])
        seed = int(normalized["seed"])
        run_id = str(normalized["run_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"corrupted normalized execution parameters: {run_dir}") from exc
    if run_id != manifest.get("run_id") or run_dir.name != run_id:
        raise ValueError(f"run id mismatch: {run_dir}")
    if manifest.get("requested_completed_trials_per_final_condition") != n_per_condition:
        raise ValueError(f"sample-size target mismatch: {run_dir}")

    plan_rows = read_jsonl(run_dir / "planned_trials.jsonl")
    expected = [row.to_dict() for row in plan_trials(
        run_id=run_id, seed=seed, n_per_condition=n_per_condition, model_configs=configs,
    )]
    if plan_rows != expected:
        raise ValueError(f"planned trials differ from deterministic balanced plan: {run_dir}")
    plans = {row["trial_id"]: PlannedTrial(**row) for row in plan_rows}
    _require_unique(plan_rows, "trial_id", "planned trial", run_dir)

    messages = read_jsonl(run_dir / "message_events.jsonl")
    assignments = read_jsonl(run_dir / "assignments.jsonl")
    attempts = read_jsonl(run_dir / "api_attempts.jsonl")
    trials = read_jsonl(run_dir / "trials.jsonl")
    _require_unique(messages, "trial_id", "message trial", run_dir)
    _require_unique(messages, "message_event_id", "message event", run_dir)
    _require_unique(assignments, "trial_id", "assignment trial", run_dir)
    _require_unique(assignments, "assignment_id", "assignment", run_dir)
    _require_unique(attempts, "attempt_id", "API attempt", run_dir)
    _require_unique(trials, "trial_id", "terminal trial", run_dir)
    attempt_keys = [
        (row.get("trial_id"), row.get("stage"), row.get("attempt_number"))
        for row in attempts
    ]
    if len(attempt_keys) != len(set(attempt_keys)):
        raise ValueError(f"duplicate trial/stage/attempt number: {run_dir}")

    messages_by_id = {row["message_event_id"]: row for row in messages}
    assignments_by_trial = {row["trial_id"]: row for row in assignments}
    for row in messages:
        plan = _known_plan(plans, row.get("trial_id"), "message", run_dir)
        if plan.planned_regime != "voluntary" or row.get("message_recorded") is not True:
            raise ValueError(f"invalid voluntary message event: {run_dir}")
    for row in assignments:
        plan = _known_plan(plans, row.get("trial_id"), "assignment", run_dir)
        try:
            condition = Condition(row["condition"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid assignment condition: {run_dir}") from exc
        if plan.planned_regime == "voluntary":
            if condition not in {
                Condition.VOLUNTARY_PROMISE_NO_SWITCH,
                Condition.VOLUNTARY_PROMISE_SWITCH,
            }:
                raise ValueError(f"voluntary trial assigned outside voluntary branches: {run_dir}")
            event = messages_by_id.get(row.get("message_event_id"))
            if event is None or event.get("trial_id") != plan.trial_id:
                raise ValueError(f"voluntary assignment precedes durable matching message: {run_dir}")
        elif condition.value != plan.final_condition or row.get("message_event_id") is not None:
            raise ValueError(f"fixed assignment differs from frozen plan: {run_dir}")
        metadata = treatment_metadata(condition, plan.trial_id)
        if (
            row.get("original_partner_id") != metadata.original_partner_id
            or row.get("final_partner_id") != metadata.final_partner_id
            or row.get("switched") != metadata.switched
        ):
            raise ValueError(f"assignment partner/switch metadata mismatch: {run_dir}")

    forbidden_labels = [condition.value for condition in Condition]
    forbidden_labels += [scientific_condition_label(condition) for condition in Condition]
    for row in attempts:
        plan = _known_plan(plans, row.get("trial_id"), "API attempt", run_dir)
        stage = row.get("stage")
        if stage not in {"message", "final_decision"}:
            raise ValueError(f"invalid API-attempt stage: {run_dir}")
        if stage == "message" and plan.planned_regime != "voluntary":
            raise ValueError(f"message attempt attached to fixed trial: {run_dir}")
        if stage == "final_decision" and plan.trial_id not in assignments_by_trial:
            raise ValueError(f"final attempt has no durable assignment: {run_dir}")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"attempt prompt-version mismatch: {run_dir}")
        prompt_text = json.dumps(
            row.get("request", {}).get("body", {}).get("messages", []),
            ensure_ascii=False,
        ).casefold()
        if any(label.casefold() in prompt_text for label in forbidden_labels):
            raise ValueError(f"researcher condition label leaked into prompt: {run_dir}")

    for row in trials:
        plan = _known_plan(plans, row.get("trial_id"), "terminal trial", run_dir)
        if row.get("run_id") != run_id:
            raise ValueError(f"terminal trial run-id mismatch: {run_dir}")
        if row.get("experiment_version") != EXPERIMENT_VERSION:
            raise ValueError(f"terminal trial experiment-version mismatch: {run_dir}")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"terminal trial prompt-version mismatch: {run_dir}")
        assignment = assignments_by_trial.get(plan.trial_id)
        if assignment and row.get("condition") != assignment.get("condition"):
            raise ValueError(f"terminal trial condition differs from assignment: {run_dir}")

    state_path = run_dir / "run_state.json"
    if not state_path.is_file():
        raise ValueError(f"missing run state: {run_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("execution_schema_version") != EXPECTED_EXECUTION_SCHEMA:
        raise ValueError(f"run-state schema mismatch: {run_dir}")
    if state.get("run_id") != run_id or set(state.get("trials", {})) != set(plans):
        raise ValueError(f"corrupted or incomplete run state: {run_dir}")
    if require_completed:
        if state.get("run_status") != "COMPLETED" or len(trials) != len(plans):
            raise ValueError(f"run is not complete and cannot enter analysis: {run_dir}")
    return manifest


def _require_unique(
    rows: list[dict[str, Any]], key: str, label: str, run_dir: Path,
) -> None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values) or len(values) != len(set(values)):
        raise ValueError(f"duplicate or missing {label} id: {run_dir}")


def _known_plan(
    plans: dict[str, Any], trial_id: Any, label: str, run_dir: Path,
) -> Any:
    try:
        return plans[str(trial_id)]
    except KeyError as exc:
        raise ValueError(f"{label} references unknown trial: {run_dir}") from exc
