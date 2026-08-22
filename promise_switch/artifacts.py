"""Read-only validation for Promise Switch execution artifact directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .design import Condition, scientific_condition_label, treatment_metadata
from .parser import parse_decision
from .randomization import allocate_voluntary_pool
from .schemas import MessageEvent, ModelConfig
from .version import (
    EXPERIMENT_VERSION, KNOWN_RESPONSE_QC_VERSIONS, PROMPT_VERSION,
    RESPONSE_QC_VERSION, prompt_templates_hash,
)

EXPECTED_EXECUTION_SCHEMA = "promise_switch_execution_v3"
EXPECTED_EXPERIMENT_VERSION = "promise_switch_v1"
LEGACY_EXECUTION_SCHEMA = "promise_switch_execution_v1"
LEGACY_EXECUTION_SCHEMAS = {LEGACY_EXECUTION_SCHEMA, "promise_switch_execution_v2"}
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
        if not (allow_legacy and schema in LEGACY_EXECUTION_SCHEMAS):
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
    response_qc_version = manifest.get("response_qc_version")
    if response_qc_version not in {None, *KNOWN_RESPONSE_QC_VERSIONS}:
        raise ValueError(f"unknown response-QC version: {run_dir}")
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
    policy = manifest.get("completion_policy")
    if not isinstance(policy, dict) or policy.get("version") != "minimum_completed_binary_v1":
        raise ValueError(f"missing or incompatible completion policy: {run_dir}")
    if role == "confirmatory" and (
        policy.get("minimum_completed_binary_per_condition") != 100
        or policy.get("top_up_enabled") is not True
    ):
        raise ValueError(f"confirmatory completed-cell policy mismatch: {run_dir}")
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
    from .execution.planner import (
        PlannedTrial, model_config_id, plan_top_up_expansion, plan_trials,
        plan_voluntary_message_recovery,
    )

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
    policy = manifest["completion_policy"]
    if (
        policy.get("minimum_completed_binary_per_condition") != n_per_condition
        or bool(normalized.get("top_up_enabled")) != bool(manifest.get("top_up_enabled"))
        or bool(policy.get("top_up_enabled")) != bool(manifest.get("top_up_enabled"))
    ):
        raise ValueError(f"top-up parameter mismatch: {run_dir}")

    plan_rows = read_jsonl(run_dir / "planned_trials.jsonl")
    initial_expected = [row.to_dict() for row in plan_trials(
        run_id=run_id, seed=seed, n_per_condition=n_per_condition, model_configs=configs,
    )]
    if plan_rows[:len(initial_expected)] != initial_expected:
        raise ValueError(f"initial trials differ from deterministic balanced plan: {run_dir}")
    top_up_events = read_jsonl(run_dir / "top_up_events.jsonl")
    _require_unique(top_up_events, "top_up_event_id", "top-up event", run_dir)
    pause_events = read_jsonl(run_dir / "pause_events.jsonl")
    _require_unique(pause_events, "pause_event_id", "pause event", run_dir)
    parser_reviews = read_jsonl(run_dir / "parser_review_events.jsonl")
    _require_unique(
        parser_reviews, "parser_review_event_id", "parser review event", run_dir,
    )
    expected_top_ups = [row for event in top_up_events for row in event.get("planned_trials", [])]
    if plan_rows[len(initial_expected):] != expected_top_ups:
        raise ValueError(f"top-up plans differ from durable expansion events: {run_dir}")
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
    attempts_by_key = {key: row for key, row in zip(attempt_keys, attempts, strict=True)}

    config_by_id = {model_config_id(config): config for config in configs}
    reviewed_keys: set[tuple[str, str]] = set()
    for review in parser_reviews:
        config_id = str(review.get("model_config_id"))
        failure_class = review.get("parser_failure_class")
        review_scope = review.get("review_scope", "class")
        key = (config_id, str(failure_class))
        if (
            manifest.get("response_qc_version") not in KNOWN_RESPONSE_QC_VERSIONS
            or config_id not in config_by_id
            or not isinstance(failure_class, str)
            or review.get("run_id") != run_id
            or review.get("review_basis") != "explicit_resume_approval"
            or review_scope not in {"class", "occurrence"}
            or (review_scope == "class" and key in reviewed_keys)
            or (
                review_scope == "occurrence"
                and (
                    manifest.get("response_qc_version") != RESPONSE_QC_VERSION
                    or not failure_class.endswith(":other_malformed_json")
                )
            )
            or (
                review_scope == "class"
                and manifest.get("response_qc_version") == RESPONSE_QC_VERSION
                and failure_class.endswith(":other_malformed_json")
            )
        ):
            raise ValueError(f"invalid parser review event: {run_dir}")
        matching_pauses = [
            event for event in pause_events
            if event.get("run_status") == "PAUSED_BEHAVIORAL_PARSER_REVIEW"
            and event.get("detail", {}).get("model_config_id") == config_id
            and event.get("detail", {}).get("parser_failure_class") == failure_class
            and event.get("detail", {}).get("trial_id") == review.get("reviewed_trial_id")
            and event.get("detail", {}).get("stage") == review.get("reviewed_stage")
            and event.get("detail", {}).get("failure_reason") == review.get("failure_reason")
            and event.get("pause_event_id") == review.get("reviewed_pause_event_id")
        ]
        if len(matching_pauses) != 1:
            raise ValueError(f"parser review has no unique matching pause: {run_dir}")
        if review_scope == "class":
            reviewed_keys.add(key)
    prior_plans = list(initial_expected)
    sequence_by_config: dict[str, int] = {}
    trials_by_id = {row.get("trial_id"): row for row in trials}
    messages_by_trial = {row.get("trial_id"): row for row in messages}
    for event in top_up_events:
        config_id = str(event.get("config_id"))
        config = config_by_id.get(config_id)
        sequence = sequence_by_config.get(config_id, 0) + 1
        if config is None or event.get("sequence") != sequence:
            raise ValueError(f"invalid top-up configuration or sequence: {run_dir}")
        if (
            event.get("completion_policy_version") != policy.get("version")
            or event.get("minimum_completed_binary_target") != n_per_condition
        ):
            raise ValueError(f"top-up event completion policy mismatch: {run_dir}")
        if event.get("prior_plan_count") != len(prior_plans):
            raise ValueError(f"invalid top-up creation order: {run_dir}")
        prior_config = [row for row in prior_plans if row["config_id"] == config_id]
        if event.get("event_type") == "voluntary_message_recovery":
            group_id = str(event.get("allocation_group_id"))
            group = [row for row in prior_config if row.get("allocation_group_id") == group_id]
            required = sum(row.get("plan_phase") == "initial" for row in group) or 2
            recorded = sum(row["trial_id"] in messages_by_trial for row in group)
            failed = [row for row in group if row["trial_id"] not in messages_by_trial
                      and trials_by_id.get(row["trial_id"], {}).get("failure_category") == "infrastructure"]
            count = required - recorded
            if (
                count < 1 or not failed
                or event.get("required_message_complete_candidates") != required
                or event.get("recorded_message_candidates_before") != recorded
                or event.get("message_recovery_count") != count
            ):
                raise ValueError(f"invalid voluntary message recovery decision: {run_dir}")
            generated = [row.to_dict() for row in plan_voluntary_message_recovery(
                run_id=run_id, seed=seed, config=config, sequence=sequence,
                allocation_group_id=group_id, count=count,
                starting_ordinal=max(row["ordinal"] for row in prior_plans),
            )]
        elif event.get("event_type") == "completed_cell_top_up":
            if any(row["trial_id"] not in trials_by_id for row in prior_config):
                raise ValueError(f"top-up created before prior trials were terminal: {run_dir}")
            counts = {condition.value: 0 for condition in Condition}
            for plan in prior_config:
                trial = trials_by_id[plan["trial_id"]]
                condition = trial.get("condition")
                if trial.get("completion_status") == "completed" and condition in counts:
                    counts[str(condition)] += 1
            fixed_deficits = {
                condition.value: max(0, n_per_condition - counts[condition.value])
                for condition in Condition if condition not in {
                    Condition.VOLUNTARY_PROMISE_NO_SWITCH,
                    Condition.VOLUNTARY_PROMISE_SWITCH,
                }
            }
            voluntary_blocks = max(
                0,
                n_per_condition - counts[Condition.VOLUNTARY_PROMISE_NO_SWITCH.value],
                n_per_condition - counts[Condition.VOLUNTARY_PROMISE_SWITCH.value],
            )
            if (
                event.get("completed_binary_counts_before") != counts
                or event.get("fixed_condition_deficits") != dict(sorted(fixed_deficits.items()))
                or event.get("voluntary_balanced_blocks") != voluntary_blocks
            ):
                raise ValueError(f"top-up decision differs from completion-only stopping rule: {run_dir}")
            generated = [row.to_dict() for row in plan_top_up_expansion(
                run_id=run_id, seed=seed, config=config, sequence=sequence,
                fixed_deficits=fixed_deficits, voluntary_blocks=voluntary_blocks,
                starting_ordinal=max(row["ordinal"] for row in prior_plans),
            )]
        else:
            raise ValueError(f"unknown top-up event type: {run_dir}")
        if event.get("planned_trials") != generated:
            raise ValueError(f"top-up identities differ from deterministic expansion: {run_dir}")
        prior_plans.extend(generated)
        sequence_by_config[config_id] = sequence

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

    voluntary_groups = sorted({
        plan.allocation_group_id for plan in plans.values()
        if plan.planned_regime == "voluntary"
    })
    for group_id in voluntary_groups:
        group = [plan for plan in plans.values() if plan.allocation_group_id == group_id]
        eligible = [plan for plan in group if plan.trial_id in messages_by_trial]
        assigned = [plan for plan in group if plan.trial_id in assignments_by_trial]
        if not assigned:
            continue
        required = sum(plan.plan_phase == "initial" for plan in group) or 2
        if len(eligible) != required or len(assigned) > required:
            raise ValueError(f"incomplete or overfilled voluntary allocation group: {run_dir}")
        events = [MessageEvent(
            message_event_id=messages_by_trial[plan.trial_id]["message_event_id"],
            trial_id=plan.trial_id,
            timestamp=messages_by_trial[plan.trial_id]["timestamp"],
            raw_message=messages_by_trial[plan.trial_id]["raw_message"],
            message_recorded=True,
        ) for plan in eligible]
        config_id = group[0].config_id
        allocation_seed = int(hashlib.sha256(
            f"{seed}|{config_id}|voluntary|{group_id}".encode()
        ).hexdigest()[:16], 16)
        expected_assignments = {
            row.trial_id: row.to_dict()
            for row in allocate_voluntary_pool(seed=allocation_seed, message_events=events)
        }
        if any(assignments_by_trial[plan.trial_id] != expected_assignments[plan.trial_id]
               for plan in assigned):
            raise ValueError(f"voluntary assignment differs from seeded message-blind block: {run_dir}")

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
        if row.get("completion_status") == "completed":
            key = (plan.trial_id, "final_decision", row.get("attempt_number"))
            attempt = attempts_by_key.get(key)
            if attempt is None or attempt.get("transport_category") != "success":
                raise ValueError(f"completed trial has no successful final attempt: {run_dir}")
            raw = attempt.get("response", {}).get("raw_response")
            if not isinstance(raw, str):
                raise ValueError(f"completed trial has malformed final response: {run_dir}")
            parsed = parse_decision(raw)
            if (
                not parsed.valid
                or row.get("final_action") != parsed.decision
                or row.get("raw_final_response") != raw
            ):
                raise ValueError(
                    f"completed trial differs from mechanically parsed final response: {run_dir}"
                )

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
        if manifest.get("dataset_role") == "confirmatory":
            counts = completed_binary_counts(trials)
            for config in configs:
                key = (config.provider, config.model_id, config.reasoning_mode)
                for condition in Condition:
                    if counts.get((*key, condition.value), 0) < n_per_condition:
                        raise ValueError(
                            "confirmatory collection has fewer than the minimum completed "
                            f"binary target in {key}|{condition.value}: {run_dir}"
                        )
    return manifest


def completed_binary_counts(
    trials: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], int]:
    """Operational completion counts; action direction is deliberately absent."""
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in trials:
        condition = row.get("condition")
        if row.get("completion_status") != "completed" or not isinstance(condition, str):
            continue
        key = (
            str(row.get("provider")), str(row.get("model")),
            str(row.get("reasoning_mode")), condition,
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


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
