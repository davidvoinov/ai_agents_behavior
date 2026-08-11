#!/usr/bin/env python3
"""Summarize Promise Switch journals without making provider calls."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from promise_switch.artifacts import read_jsonl, validate_run_directory
from promise_switch.design import Condition, scientific_condition_label, treatment_metadata
from promise_switch.parser import parse_decision
from promise_switch.prompts import final_decision_messages


def numeric_summary(values: list[float | int]) -> dict[str, float | int | None]:
    return ({"count": 0, "sum": 0, "min": None, "median": None, "max": None}
            if not values else {"count": len(values), "sum": sum(values), "min": min(values),
                                "median": statistics.median(values), "max": max(values)})


def audit(
    run_dir: Path, *, include_invalid: bool = False, include_legacy: bool = False,
) -> dict[str, Any]:
    manifest = validate_run_directory(
        run_dir, allow_invalid=include_invalid, allow_legacy=include_legacy,
    )
    plans = read_jsonl(run_dir / "planned_trials.jsonl")
    messages = read_jsonl(run_dir / "message_events.jsonl")
    assignments = read_jsonl(run_dir / "assignments.jsonl")
    attempts = read_jsonl(run_dir / "api_attempts.jsonl")
    trials = read_jsonl(run_dir / "trials.jsonl")
    errors = read_jsonl(run_dir / "errors.jsonl")
    completion_tokens = [row["response"]["usage"]["completion_tokens"] for row in attempts
                         if row.get("response", {}).get("usage", {}).get("completion_tokens") is not None]
    prompt_tokens = [row["response"]["usage"]["prompt_tokens"] for row in attempts
                     if row.get("response", {}).get("usage", {}).get("prompt_tokens") is not None]
    latency = [row["response"]["latency_seconds"] for row in attempts
               if row.get("response", {}).get("latency_seconds") is not None]
    assignment_counts = Counter(row.get("condition") for row in assignments)
    trial_counts = Counter(row.get("condition") for row in trials)
    assignments_by_trial = {row["trial_id"]: row for row in assignments}
    messages_by_id = {row["message_event_id"]: row for row in messages}
    final_attempts = [row for row in attempts if row.get("stage") == "final_decision"
                      and row.get("response", {}).get("status") == "success"
                      and row.get("response", {}).get("finish_reason") == "stop"]
    final_prompt_matches = 0
    valid_final_parses = 0
    partner_metadata_matches = 0
    for attempt in final_attempts:
        assignment = assignments_by_trial.get(attempt["trial_id"])
        if not assignment:
            continue
        raw_message = ""
        if assignment.get("message_event_id") in messages_by_id:
            raw_message = messages_by_id[assignment["message_event_id"]].get("raw_message", "")
        expected = final_decision_messages(
            condition=Condition(assignment["condition"]),
            original_partner_id=assignment["original_partner_id"],
            final_partner_id=assignment["final_partner_id"],
            raw_message=raw_message,
        )
        final_prompt_matches += attempt.get("request", {}).get("body", {}).get("messages") == expected
        valid_final_parses += parse_decision(attempt["response"].get("raw_response", "")).valid
    for assignment in assignments:
        expected = treatment_metadata(Condition(assignment["condition"]), assignment["trial_id"])
        partner_metadata_matches += (
            assignment["original_partner_id"] == expected.original_partner_id
            and assignment["final_partner_id"] == expected.final_partner_id
            and assignment["switched"] == expected.switched
        )
    return {
        "run_id": manifest["run_id"],
        "dataset_role": manifest.get("dataset_role", "legacy_unclassified_development"),
        "invalid_concurrent_execution": (run_dir / "INVALID_CONCURRENT_EXECUTION.json").exists(),
        "model_configurations": manifest["model_configurations"],
        "condition_labels": {
            condition.value: scientific_condition_label(condition)
            for condition in Condition
        },
        "temperature": manifest["temperature"],
        "token_limits": manifest["token_limits"],
        "rows": {"planned": len(plans), "messages": len(messages), "assignments": len(assignments),
                 "attempts": len(attempts), "trials": len(trials), "errors": len(errors)},
        "unique_ids": {"planned_trials": len({row["trial_id"] for row in plans}),
                       "message_events": len({row["message_event_id"] for row in messages}),
                       "assignments": len({row["assignment_id"] for row in assignments}),
                       "attempts": len({row["attempt_id"] for row in attempts}),
                       "trials": len({row["trial_id"] for row in trials})},
        "assignment_condition_counts": dict(sorted(assignment_counts.items())),
        "trial_condition_counts": dict(sorted(trial_counts.items())),
        "final_actions": dict(sorted(Counter(row.get("final_action") for row in trials).items())),
        "attempt_numbers": dict(sorted(Counter(row.get("attempt_number") for row in attempts).items())),
        "attempt_stages": dict(sorted(Counter(row.get("stage") for row in attempts).items())),
        "attempt_statuses": dict(sorted(Counter(row.get("response", {}).get("status") for row in attempts).items())),
        "finish_reasons": dict(sorted(Counter(row.get("response", {}).get("finish_reason") for row in attempts).items())),
        "transport_categories": dict(sorted(Counter(row.get("transport_category") for row in attempts).items())),
        "message_parse": {"valid": sum(row.get("parse_valid") is True for row in messages),
                          "invalid": sum(row.get("parse_valid") is False for row in messages)},
        "trial_completion": dict(sorted(Counter(row.get("completion_status") for row in trials).items())),
        "trial_failures": dict(sorted(Counter(str(row.get("failure_category")) + ":" + str(row.get("failure_reason"))
                                               for row in trials if row.get("failure_category")).items())),
        "voluntary_messages": dict(sorted(Counter(row.get("raw_message", "") for row in messages).items())),
        "mechanics": {
            "all_six_conditions_present": set(assignment_counts) == {condition.value for condition in Condition},
            "exact_condition_balance": len(set(assignment_counts.values())) == 1,
            "all_recorded_messages_linked": all(row.get("message_event_id") in messages_by_id
                                                   for row in assignments
                                                   if row.get("condition", "").startswith("voluntary_")),
            "partner_metadata_matches": f"{partner_metadata_matches}/{len(assignments)}",
            "successful_final_prompts_match_frozen_templates": f"{final_prompt_matches}/{len(final_attempts)}",
            "successful_final_responses_parse": f"{valid_final_parses}/{len(final_attempts)}",
        },
        "completion_tokens": numeric_summary(completion_tokens),
        "prompt_tokens": numeric_summary(prompt_tokens),
        "latency_seconds": numeric_summary(latency),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--include-invalid", action="store_true",
        help="permit explicit technical inspection of runs marked invalid",
    )
    parser.add_argument(
        "--include-legacy", action="store_true",
        help="permit explicit technical inspection of pre-role execution schemas",
    )
    args = parser.parse_args()
    resolved = [run_dir.resolve() for run_dir in args.run_dirs]
    if len(resolved) != len(set(resolved)):
        raise SystemExit("A run directory may be supplied only once")
    try:
        reports = [audit(
            run_dir,
            include_invalid=args.include_invalid,
            include_legacy=args.include_legacy,
        )
                   for run_dir in sorted(resolved)]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
