"""Crash-safe production execution for the frozen Promise Switch design."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from ..artifacts import validate_run_integrity
from ..client import CompletionTransport, ProviderAgnosticClient, RequestSpec
from ..design import Condition, scientific_condition_label, treatment_metadata
from ..parser import parse_decision, parse_message
from ..prompts import final_decision_messages, voluntary_message_messages
from ..randomization import allocate_voluntary_pool, assign_fixed_trial, plan_fixed_trial
from ..schemas import (
    AssignmentRecord,
    MessageEvent,
    ModelConfig,
    NormalizedResponse,
    sanitize_for_serialization,
)
from ..version import EXPERIMENT_VERSION, PROMPT_VERSION, prompt_templates_hash
from .attrition import (
    AttritionOutcome,
    classify_decision_output,
    classify_message_output,
    classify_transport,
)
from .lock import RunWriterLock
from .planner import (
    PlannedTrial, model_config_id, plan_top_up_expansion,
    plan_trials, plan_voluntary_message_recovery,
)
from .rate_limit import RateLimiter, RateLimitPolicy
from .retry import RetryPolicy
from .scheduler import balanced_pending_order
from .state import RunStatus, TrialState, is_terminal, transition
from .storage import RunStorage
from .transport import OpenAICompatibleTransport

EXECUTION_SCHEMA_VERSION = "promise_switch_execution_v3"
COMPLETION_POLICY_VERSION = "minimum_completed_binary_v1"
BEHAVIORAL_PARSER_FAILURE_REVIEW_COUNT = 2
DatasetRole = Literal["technical", "pilot", "confirmatory", "synthetic"]
Hook = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class ExecutionParameters:
    run_id: str
    output_dir: Path
    n_per_condition: int
    seed: int
    model_configs: tuple[ModelConfig, ...]
    dataset_role: DatasetRole
    temperature: float = 1.0
    temperature_override_authorized: bool = False
    message_max_tokens: int | None = None
    decision_max_tokens: int | None = None
    output_cap_override_authorized: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit_policy: RateLimitPolicy = field(default_factory=RateLimitPolicy)
    dry_run: bool = False
    top_up_enabled: bool = False
    cli_invocation: tuple[str, ...] = ()

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def __post_init__(self) -> None:
        if not self.run_id.strip() or "/" in self.run_id or self.run_id in {".", ".."}:
            raise ValueError("run_id must be a non-empty path-safe name")
        if self.n_per_condition < 1:
            raise ValueError("n_per_condition must be positive")
        if not self.model_configs:
            raise ValueError("At least one model configuration is required")
        if self.dataset_role not in {"technical", "pilot", "confirmatory", "synthetic"}:
            raise ValueError("dataset_role must explicitly classify every run")
        if self.dry_run != (self.dataset_role == "synthetic"):
            raise ValueError("dry_run must be true exactly for synthetic runs")
        if self.dataset_role == "confirmatory" and self.n_per_condition != 100:
            raise ValueError("confirmatory runs require the approved completed-cell target n=100")
        if self.dataset_role == "confirmatory" and not self.top_up_enabled:
            object.__setattr__(self, "top_up_enabled", True)
        if self.temperature != 1.0 and not self.temperature_override_authorized:
            raise ValueError(
                "experimental-agent temperature differs from the frozen production value "
                "1.0; an explicit authorized override is required"
            )
        validated_caps = {
            config.production_max_output_tokens for config in self.model_configs
        }
        if len(validated_caps) != 1:
            raise ValueError(
                "one run may contain only model configurations with the same validated "
                "production output cap"
            )
        validated_cap = validated_caps.pop()
        decision_cap = (
            validated_cap if self.decision_max_tokens is None else self.decision_max_tokens
        )
        message_cap = (
            decision_cap if self.message_max_tokens is None else self.message_max_tokens
        )
        if message_cap < 1 or decision_cap < 1:
            raise ValueError("output-token caps must be positive")
        if (
            (message_cap != validated_cap or decision_cap != validated_cap)
            and not self.output_cap_override_authorized
        ):
            raise ValueError(
                "output-token cap differs from the validated production cap; "
                "an explicit authorized override is required"
            )
        object.__setattr__(self, "message_max_tokens", message_cap)
        object.__setattr__(self, "decision_max_tokens", decision_cap)


class DeterministicDryRunTransport(CompletionTransport):
    """Network-free transport that exercises the complete execution path."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.calls += 1
        stage = request.metadata.get("stage")
        if stage == "message":
            raw = json.dumps({"message": f"dry-run-message-{self.calls}"})
        else:
            raw = json.dumps({"decision": "COOPERATE", "justification": "Dry run."})
        return NormalizedResponse(
            status="success", raw_response=raw,
            provider_response_id=f"dry-response-{self.calls}",
            provider_request_id=f"dry-request-{self.calls}",
            http_status=200, finish_reason="stop", latency_seconds=0.0,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )


class PromiseSwitchRunner:
    def __init__(
        self,
        parameters: ExecutionParameters,
        *,
        transport: CompletionTransport | Mapping[str, CompletionTransport] | None = None,
        retry_sleeper: Callable[[float], None] = time.sleep,
        rate_limiter_factory: Callable[[ModelConfig], RateLimiter] | None = None,
        retry_random: Callable[[], float] = lambda: 0.5,
        hook: Hook | None = None,
        timestamp: Callable[[], str] | None = None,
    ):
        self.parameters = parameters
        self.storage = RunStorage(parameters.run_dir)
        self.transport = transport
        self.retry_sleeper = retry_sleeper
        self.rate_limiter_factory = rate_limiter_factory
        self.retry_random = retry_random
        self.hook = hook or (lambda _event, _payload: None)
        self.timestamp = timestamp or _utc_now
        self.plans: tuple[PlannedTrial, ...] = ()
        self.states: dict[str, TrialState] = {}
        self.assignments: dict[str, dict[str, Any]] = {}
        self.message_events: dict[str, dict[str, Any]] = {}
        self.trials: dict[str, dict[str, Any]] = {}
        self.api_attempts: list[dict[str, Any]] = []
        self.run_status = RunStatus.CREATED
        self.pause_detail: dict[str, Any] | None = None
        self.manifest: dict[str, Any] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._configs = {model_config_id(config): config for config in parameters.model_configs}
        self._default_dry_transport = DeterministicDryRunTransport()

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        with RunWriterLock(
            self.parameters.run_dir, run_id=self.parameters.run_id,
        ):
            return self._run_locked(resume=resume)

    def _run_locked(self, *, resume: bool) -> dict[str, Any]:
        if resume:
            self._load_for_resume()
        else:
            self._create_run()
        self.run_status = RunStatus.RUNNING
        self.pause_detail = None
        self._reconcile()
        self._persist_state()

        while True:
            allocated = self._allocate_ready_voluntary_pools()
            recovered = self._append_required_message_recovery_trials()
            conditions = {
                plan.trial_id: self.assignments.get(plan.trial_id, {}).get("condition")
                or plan.final_condition
                for plan in self.plans
            }
            order = balanced_pending_order(self.plans, self.states, conditions)
            progressed = allocated or recovered
            for trial_id in order:
                if is_terminal(self.states[trial_id]):
                    continue
                if self._process_trial(self._plan(trial_id)):
                    progressed = True
                if self.run_status in {
                    RunStatus.PAUSED_PROVIDER_QUOTA,
                    RunStatus.PAUSED_INFRASTRUCTURE_REVIEW,
                    RunStatus.PAUSED_BEHAVIORAL_PARSER_REVIEW,
                }:
                    self._persist_state()
                    return self.technical_summary()
            if all(is_terminal(state) for state in self.states.values()):
                if self.parameters.top_up_enabled and not self._completion_target_met():
                    if self._append_required_top_ups():
                        continue
                    self.run_status = RunStatus.FAILED
                    self._persist_state()
                    return self.technical_summary()
                self.run_status = RunStatus.COMPLETED
                self._persist_state()
                return self.technical_summary()
            if not progressed:
                self.run_status = RunStatus.FAILED
                self._persist_state()
                return self.technical_summary()

    def preflight_estimate(self) -> dict[str, Any]:
        configs = []
        for config in self.parameters.model_configs:
            n = self.parameters.n_per_condition
            configs.append({
                "provider": config.provider,
                "model": config.model_id,
                "reasoning_mode": config.reasoning_mode,
                "validated_production_max_output_tokens": (
                    config.production_max_output_tokens
                ),
                "target_final_cells": {condition.value: n for condition in Condition},
                "planned_trials": 6 * n,
                "approximate_api_calls": 8 * n,
                "approximate_message_stage_calls": 2 * n,
                "approximate_decision_stage_calls": 6 * n,
                "estimated_tokens": "unknown",
                "estimated_cost": "unknown",
            })
        return {
            "run_id": self.parameters.run_id,
            "dataset_role": self.parameters.dataset_role,
            "temperature": self.parameters.temperature,
            "temperature_override_authorized": (
                self.parameters.temperature_override_authorized
            ),
            "token_limits": {
                "message": self.parameters.message_max_tokens,
                "final_decision": self.parameters.decision_max_tokens,
            },
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_templates_hash(),
            "completion_policy": {
                "version": COMPLETION_POLICY_VERSION,
                "minimum_completed_binary_per_condition": self.parameters.n_per_condition,
                "top_up_enabled": self.parameters.top_up_enabled,
                "voluntary_top_up_unit": "randomized_balanced_block_of_2",
            },
            "model_configurations": configs,
        }

    def technical_summary(self) -> dict[str, Any]:
        completed_actions = sum(
            1 for row in self.trials.values()
            if row.get("completion_status") == "completed" and row.get("final_action")
        )
        infrastructure = sum(
            1 for row in self.trials.values() if row.get("failure_category") == "infrastructure"
        )
        behavioral = sum(
            1 for row in self.trials.values() if row.get("failure_category") == "behavioral"
        )
        parser_failures = self._behavioral_parser_failure_counts()
        retries_by_trial: dict[str, int] = {}
        for attempt in self.api_attempts:
            if int(attempt.get("attempt_number", 0)) > 1:
                trial_id = str(attempt.get("trial_id"))
                retries_by_trial[trial_id] = retries_by_trial.get(trial_id, 0) + 1
        cells: dict[str, dict[str, int]] = {}
        for plan in self.plans:
            assignment = self.assignments.get(plan.trial_id)
            condition = (assignment or {}).get("condition") or plan.final_condition or "voluntary_pending"
            key = "|".join((plan.provider, plan.model, plan.reasoning_mode, condition))
            cell = cells.setdefault(key, {
                "initiated": 0, "completed_binary": 0, "behavioral_missing_action": 0,
                "terminal_infrastructure_attrition": 0, "pending": 0,
                "top_up_trials_added": 0, "retry_attempts": 0,
            })
            cell["initiated"] += 1
            cell["top_up_trials_added"] += plan.plan_phase == "top_up"
            cell["retry_attempts"] += retries_by_trial.get(plan.trial_id, 0)
            row = self.trials.get(plan.trial_id)
            if not row:
                cell["pending"] += 1
            elif row.get("completion_status") == "completed":
                cell["completed_binary"] += 1
            elif row.get("failure_category") == "behavioral":
                cell["behavioral_missing_action"] += 1
            else:
                cell["terminal_infrastructure_attrition"] += 1
        return {
            "run_id": self.parameters.run_id,
            "run_status": self.run_status.value,
            "planned_trials": len(self.plans),
            "initial_planned_trials": sum(plan.plan_phase == "initial" for plan in self.plans),
            "top_up_trials_added": sum(plan.plan_phase == "top_up" for plan in self.plans),
            "top_up_events": len(self.storage.read_jsonl("top_up_events.jsonl")),
            "pause_events": len(self.storage.read_jsonl("pause_events.jsonl")),
            "minimum_completed_binary_target": self.parameters.n_per_condition,
            "completion_target_met": self._completion_target_met(),
            "completed_final_actions": completed_actions,
            "pending": sum(not is_terminal(state) for state in self.states.values()),
            "infrastructure_failures": infrastructure,
            "behavioral_failures": behavioral,
            "behavioral_parser_failures_by_model_configuration": parser_failures,
            "behavioral_parser_failure_review_required": any(
                row["total"] >= BEHAVIORAL_PARSER_FAILURE_REVIEW_COUNT
                for row in parser_failures.values()
            ),
            "paused_quota": self.run_status == RunStatus.PAUSED_PROVIDER_QUOTA,
            "paused_infrastructure_review": (
                self.run_status == RunStatus.PAUSED_INFRASTRUCTURE_REVIEW
            ),
            "paused_behavioral_parser_review": (
                self.run_status == RunStatus.PAUSED_BEHAVIORAL_PARSER_REVIEW
            ),
            "pause_detail": self.pause_detail,
            "counts_by_model_provider_reasoning_condition": cells,
        }

    def _completion_counts(self, config_id: str) -> dict[str, int]:
        """Count valid completion status only; never inspect action direction."""
        plan_ids = {plan.trial_id for plan in self.plans if plan.config_id == config_id}
        return {
            condition.value: sum(
                row.get("completion_status") == "completed"
                and row.get("condition") == condition.value
                and trial_id in plan_ids
                for trial_id, row in self.trials.items()
            )
            for condition in Condition
        }

    def _completion_target_met(self) -> bool:
        if not self.plans:
            return False
        target = self.parameters.n_per_condition
        return all(
            all(count >= target for count in self._completion_counts(config_id).values())
            for config_id in self._configs
        )

    def _append_required_top_ups(self) -> bool:
        """Append only completion-status-driven expansions after current work is terminal."""
        added = False
        target = self.parameters.n_per_condition
        fixed_values = {condition.value for condition in Condition if condition not in {
            Condition.VOLUNTARY_PROMISE_NO_SWITCH,
            Condition.VOLUNTARY_PROMISE_SWITCH,
        }}
        for config_id, config in sorted(self._configs.items()):
            config_plans = [plan for plan in self.plans if plan.config_id == config_id]
            if not config_plans or not all(is_terminal(self.states[plan.trial_id]) for plan in config_plans):
                continue
            counts = self._completion_counts(config_id)
            fixed_deficits = {
                value: max(0, target - counts[value]) for value in sorted(fixed_values)
            }
            voluntary_blocks = max(
                0,
                target - counts[Condition.VOLUNTARY_PROMISE_NO_SWITCH.value],
                target - counts[Condition.VOLUNTARY_PROMISE_SWITCH.value],
            )
            if not any(fixed_deficits.values()) and not voluntary_blocks:
                continue
            prior_events = [row for row in self.storage.read_jsonl("top_up_events.jsonl")
                            if row.get("config_id") == config_id]
            sequence = len(prior_events) + 1
            plans = plan_top_up_expansion(
                run_id=self.parameters.run_id, seed=self.parameters.seed, config=config,
                sequence=sequence, fixed_deficits=fixed_deficits,
                voluntary_blocks=voluntary_blocks,
                starting_ordinal=max(plan.ordinal for plan in self.plans),
            )
            event = sanitize_for_serialization({
                "top_up_event_id": "topup-event:" + hashlib.sha256(
                    f"{self.parameters.run_id}|{config_id}|{sequence}".encode()
                ).hexdigest()[:20],
                "timestamp": self.timestamp(),
                "run_id": self.parameters.run_id,
                "config_id": config_id,
                "sequence": sequence,
                "event_type": "completed_cell_top_up",
                "completion_policy_version": COMPLETION_POLICY_VERSION,
                "minimum_completed_binary_target": target,
                "eligibility_reason": "all_current_trials_terminal_and_cell_below_target",
                "prior_plan_count": len(self.plans),
                "completed_binary_counts_before": counts,
                "fixed_condition_deficits": fixed_deficits,
                "voluntary_balanced_blocks": voluntary_blocks,
                "derivation": "seed+run_id+config_id+sequence+regime+block+slot",
                "planned_trials": [plan.to_dict() for plan in plans],
            })
            self.hook("before_top_up_creation", event)
            self.storage.append("top_up_events.jsonl", event)
            self.hook("after_top_up_event", event)
            for plan in plans:
                self.storage.append("planned_trials.jsonl", plan.to_dict())
                self.hook("after_top_up_plan_persistence", plan.to_dict())
            self.plans = (*self.plans, *plans)
            for plan in plans:
                self.states[plan.trial_id] = TrialState.PLANNED
            self._persist_state()
            added = True
        return added

    def _append_required_message_recovery_trials(self) -> bool:
        """Restore two message-complete candidates per randomized voluntary block."""
        added = False
        for config_id, config in sorted(self._configs.items()):
            groups = sorted({
                plan.allocation_group_id for plan in self.plans
                if plan.config_id == config_id and plan.planned_regime == "voluntary"
                and plan.trial_id not in self.assignments
            })
            for group_id in groups:
                group = [plan for plan in self.plans
                         if plan.config_id == config_id and plan.allocation_group_id == group_id]
                required = sum(plan.plan_phase == "initial" for plan in group) or 2
                recorded = sum(plan.trial_id in self.message_events for plan in group)
                if recorded >= required:
                    continue
                unresolved = [plan for plan in group if plan.trial_id not in self.message_events
                              and not is_terminal(self.states[plan.trial_id])]
                if unresolved:
                    continue
                failures = [plan for plan in group if plan.trial_id not in self.message_events
                            and self.states[plan.trial_id] == TrialState.FAILED_INFRASTRUCTURE_FINAL]
                if not failures:
                    continue
                count = required - recorded
                prior_events = [row for row in self.storage.read_jsonl("top_up_events.jsonl")
                                if row.get("config_id") == config_id]
                sequence = len(prior_events) + 1
                plans = plan_voluntary_message_recovery(
                    run_id=self.parameters.run_id, seed=self.parameters.seed, config=config,
                    sequence=sequence, allocation_group_id=str(group_id), count=count,
                    starting_ordinal=max(plan.ordinal for plan in self.plans),
                )
                event = sanitize_for_serialization({
                    "top_up_event_id": "topup-event:" + hashlib.sha256(
                        f"{self.parameters.run_id}|{config_id}|{sequence}".encode()
                    ).hexdigest()[:20],
                    "timestamp": self.timestamp(), "run_id": self.parameters.run_id,
                    "config_id": config_id, "sequence": sequence,
                    "event_type": "voluntary_message_recovery",
                    "completion_policy_version": COMPLETION_POLICY_VERSION,
                    "minimum_completed_binary_target": self.parameters.n_per_condition,
                    "eligibility_reason": "message_stage_infrastructure_attrition_before_assignment",
                    "prior_plan_count": len(self.plans),
                    "allocation_group_id": group_id,
                    "required_message_complete_candidates": required,
                    "recorded_message_candidates_before": recorded,
                    "message_recovery_count": count,
                    "derivation": "seed+run_id+config_id+sequence+allocation_group+slot",
                    "planned_trials": [plan.to_dict() for plan in plans],
                })
                self.storage.append("top_up_events.jsonl", event)
                self.hook("after_top_up_event", event)
                for plan in plans:
                    self.storage.append("planned_trials.jsonl", plan.to_dict())
                    self.hook("after_top_up_plan_persistence", plan.to_dict())
                self.plans = (*self.plans, *plans)
                for plan in plans:
                    self.states[plan.trial_id] = TrialState.PLANNED
                self._persist_state()
                added = True
        return added

    def _reconcile_top_up_plan_journal(self) -> None:
        """Finish an expansion whose durable event survived a process crash."""
        events = self.storage.read_jsonl("top_up_events.jsonl")
        expected = [plan for event in events for plan in event.get("planned_trials", [])]
        rows = self.storage.read_jsonl("planned_trials.jsonl")
        base_count = 6 * self.parameters.n_per_condition * len(self.parameters.model_configs)
        actual = rows[base_count:]
        if actual != expected[:len(actual)]:
            raise ValueError("top-up plan journal differs from durable expansion events")
        for row in expected[len(actual):]:
            self.storage.append("planned_trials.jsonl", row)
        state = self.storage.read_state()
        if state:
            trial_states = dict(state.get("trials", {}))
            changed = False
            for row in expected:
                if row["trial_id"] not in trial_states:
                    trial_states[row["trial_id"]] = TrialState.PLANNED.value
                    changed = True
            if changed:
                state["trials"] = trial_states
                state["updated_at"] = self.timestamp()
                self.storage.write_state(state)

    def _create_run(self) -> None:
        self.storage.initialize()
        self.plans = plan_trials(
            run_id=self.parameters.run_id, seed=self.parameters.seed,
            n_per_condition=self.parameters.n_per_condition,
            model_configs=self.parameters.model_configs,
        )
        self.manifest = self._manifest()
        self.storage.create_manifest(self.manifest)
        self.storage.append_many("planned_trials.jsonl", (plan.to_dict() for plan in self.plans))
        self.states = {plan.trial_id: TrialState.PLANNED for plan in self.plans}
        self._persist_state()

    def _load_for_resume(self) -> None:
        manifest = self.storage.read_manifest()
        if manifest.get("execution_schema_version") != EXECUTION_SCHEMA_VERSION:
            raise ValueError("Refusing to resume a legacy or incompatible Promise Switch run")
        self._reconcile_top_up_plan_journal()
        validate_run_integrity(self.parameters.run_dir)
        if manifest.get("experiment_version") != EXPERIMENT_VERSION:
            raise ValueError("Experiment version differs from immutable manifest")
        if manifest.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("Prompt version differs from immutable manifest")
        if manifest.get("prompt_hash") != prompt_templates_hash():
            raise ValueError("Current prompt hash differs from immutable manifest")
        expected = self._normalized_parameters()
        if manifest.get("normalized_execution_parameters") != expected:
            raise ValueError("Resume parameters are incompatible with immutable manifest")
        self.manifest = manifest
        self.plans = tuple(PlannedTrial(**row) for row in self.storage.read_jsonl("planned_trials.jsonl"))
        if not self.plans:
            raise ValueError("Run has no durable planned trials")

    def _reconcile(self) -> None:
        self.message_events = _last_by(self.storage.read_jsonl("message_events.jsonl"), "trial_id")
        self.assignments = _last_by(self.storage.read_jsonl("assignments.jsonl"), "trial_id")
        self.trials = _last_by(self.storage.read_jsonl("trials.jsonl"), "trial_id")
        self.api_attempts = self.storage.read_jsonl("api_attempts.jsonl")
        states: dict[str, TrialState] = {}
        for plan in self.plans:
            row = self.trials.get(plan.trial_id)
            if row:
                if row.get("completion_status") == "completed":
                    states[plan.trial_id] = TrialState.COMPLETED
                elif row.get("failure_category") == "behavioral":
                    states[plan.trial_id] = TrialState.BEHAVIORAL_FAILURE
                else:
                    states[plan.trial_id] = TrialState.FAILED_INFRASTRUCTURE_FINAL
            elif plan.trial_id in self.assignments:
                states[plan.trial_id] = TrialState.DECISION_PENDING if self._reusable_attempt(
                    plan.trial_id, "final_decision"
                ) else TrialState.ASSIGNED
            elif plan.planned_regime == "voluntary":
                states[plan.trial_id] = (
                    TrialState.ASSIGNMENT_PENDING if plan.trial_id in self.message_events
                    else TrialState.MESSAGE_PENDING
                )
            else:
                states[plan.trial_id] = TrialState.PLANNED
        self.states = states

    def _process_trial(self, plan: PlannedTrial) -> bool:
        state = self.states[plan.trial_id]
        if state == TrialState.PLANNED:
            if plan.planned_regime == "voluntary":
                self._set_state(plan.trial_id, TrialState.MESSAGE_PENDING)
            else:
                self._append_fixed_assignment(plan)
            return True
        if self.states[plan.trial_id] == TrialState.MESSAGE_PENDING:
            self._run_message_stage(plan)
            return True
        if self.states[plan.trial_id] == TrialState.ASSIGNMENT_PENDING:
            return False
        if self.states[plan.trial_id] == TrialState.ASSIGNED:
            event = self.message_events.get(plan.trial_id)
            if event and event.get("behavioral_failure_reason"):
                self._append_trial_result(
                    plan, final_action=None, raw_final_response="", completion_status="failed",
                    failure_category="behavioral",
                    failure_reason=event["behavioral_failure_reason"], attempt_number=0,
                )
                self._set_state(plan.trial_id, TrialState.BEHAVIORAL_FAILURE)
            else:
                self._set_state(plan.trial_id, TrialState.DECISION_PENDING)
            return True
        if self.states[plan.trial_id] == TrialState.DECISION_PENDING:
            self._run_decision_stage(plan)
            return True
        return False

    def _append_fixed_assignment(self, plan: PlannedTrial) -> None:
        condition = Condition(plan.final_condition or "")
        foundation_plan = plan_fixed_trial(
            trial_id=plan.trial_id, run_id=plan.run_id,
            seed=plan.randomization_seed, condition=condition,
        )
        assignment = assign_fixed_trial(foundation_plan)
        self.storage.append("assignments.jsonl", assignment.to_dict())
        self.assignments[plan.trial_id] = assignment.to_dict()
        self.hook("after_assignment", assignment.to_dict())
        self._set_state(plan.trial_id, TrialState.ASSIGNED)

    def _run_message_stage(self, plan: PlannedTrial) -> None:
        metadata = treatment_metadata(Condition.VOLUNTARY_PROMISE_NO_SWITCH, plan.trial_id)
        messages = voluntary_message_messages(metadata.original_partner_id)
        response, outcome, attempt_number = self._execute_request(
            plan, "message", messages, self.parameters.message_max_tokens,
        )
        if outcome.quota_pause or response is None:
            return
        if outcome.category == "infrastructure":
            self._append_trial_result(
                plan, final_action=None, raw_final_response="", completion_status="failed",
                failure_category="infrastructure", failure_reason=outcome.cause,
                attempt_number=attempt_number,
            )
            self._set_state(plan.trial_id, TrialState.FAILED_INFRASTRUCTURE_FINAL)
            self._pause_infrastructure_review(
                plan, "message", outcome, response, attempt_number,
            )
            return
        parsed = parse_message(response.raw_response)
        behavioral = classify_message_output(parsed)
        event_id = "message:" + hashlib.sha256(
            f"{plan.trial_id}|{response.raw_response}".encode()
        ).hexdigest()[:20]
        event = MessageEvent(
            message_event_id=event_id, trial_id=plan.trial_id,
            timestamp=self.timestamp(),
            raw_message=parsed.message if parsed.valid else response.raw_response,
            message_recorded=True,
        ).to_dict()
        event.update({
            "raw_response": response.raw_response,
            "parsed_message": parsed.message,
            "parse_valid": parsed.valid,
            "parse_error": parsed.error,
            "behavioral_failure_reason": behavioral.cause,
            "attempt_number": attempt_number,
        })
        self.storage.append("message_events.jsonl", event)
        self.message_events[plan.trial_id] = event
        self.hook("after_message_event", event)
        self._set_state(plan.trial_id, TrialState.MESSAGE_RECORDED)
        self._set_state(plan.trial_id, TrialState.ASSIGNMENT_PENDING)
        if behavioral.category == "behavioral":
            self._maybe_pause_behavioral_parser_review(
                plan, "message", behavioral, attempt_number,
            )

    def _allocate_ready_voluntary_pools(self) -> bool:
        changed = False
        for config_id in sorted(self._configs):
            groups = sorted({
                plan.allocation_group_id for plan in self.plans
                if plan.config_id == config_id and plan.planned_regime == "voluntary"
            })
            for group_id in groups:
                all_group = [plan for plan in self.plans if plan.config_id == config_id
                             and plan.allocation_group_id == group_id]
                required = sum(plan.plan_phase == "initial" for plan in all_group) or 2
                pool = [plan for plan in all_group if plan.trial_id in self.message_events]
                if len(pool) != required:
                    continue
                events = [MessageEvent(
                    message_event_id=self.message_events[plan.trial_id]["message_event_id"],
                    trial_id=plan.trial_id,
                    timestamp=self.message_events[plan.trial_id]["timestamp"],
                    raw_message=self.message_events[plan.trial_id]["raw_message"],
                    message_recorded=True,
                ) for plan in pool]
                allocation_seed = _allocation_seed(self.parameters.seed, config_id, str(group_id))
                for assignment in allocate_voluntary_pool(seed=allocation_seed, message_events=events):
                    if assignment.trial_id in self.assignments:
                        continue
                    row = assignment.to_dict()
                    self.storage.append("assignments.jsonl", row)
                    self.assignments[assignment.trial_id] = row
                    self.hook("after_assignment", row)
                    current = self.states[assignment.trial_id]
                    if current == TrialState.MESSAGE_RECORDED:
                        self._set_state(assignment.trial_id, TrialState.ASSIGNMENT_PENDING)
                    self._set_state(assignment.trial_id, TrialState.ASSIGNED)
                    changed = True
        return changed

    def _run_decision_stage(self, plan: PlannedTrial) -> None:
        assignment = self.assignments[plan.trial_id]
        condition = Condition(assignment["condition"])
        event = self.message_events.get(plan.trial_id)
        messages = final_decision_messages(
            condition=condition,
            original_partner_id=assignment["original_partner_id"],
            final_partner_id=assignment["final_partner_id"],
            raw_message=(event or {}).get("raw_message", ""),
        )
        response, outcome, attempt_number = self._execute_request(
            plan, "final_decision", messages, self.parameters.decision_max_tokens,
        )
        if outcome.quota_pause or response is None:
            return
        if outcome.category == "infrastructure":
            self._append_trial_result(
                plan, final_action=None, raw_final_response=response.raw_response,
                completion_status="failed", failure_category="infrastructure",
                failure_reason=outcome.cause, attempt_number=attempt_number,
            )
            self._set_state(plan.trial_id, TrialState.FAILED_INFRASTRUCTURE_FINAL)
            self._pause_infrastructure_review(
                plan, "final_decision", outcome, response, attempt_number,
            )
            return
        parsed = parse_decision(response.raw_response)
        behavioral = classify_decision_output(parsed)
        if behavioral.category == "behavioral":
            self._append_trial_result(
                plan, final_action=None, raw_final_response=response.raw_response,
                completion_status="failed", failure_category="behavioral",
                failure_reason=behavioral.cause, attempt_number=attempt_number,
            )
            self._set_state(plan.trial_id, TrialState.BEHAVIORAL_FAILURE)
            self._maybe_pause_behavioral_parser_review(
                plan, "final_decision", behavioral, attempt_number,
            )
            return
        self._append_trial_result(
            plan, final_action=parsed.decision, raw_final_response=response.raw_response,
            completion_status="completed", failure_category=None, failure_reason=None,
            attempt_number=attempt_number,
        )
        self._set_state(plan.trial_id, TrialState.COMPLETED)

    def _execute_request(
        self, plan: PlannedTrial, stage: str,
        messages: Sequence[dict[str, str]], max_tokens: int,
    ) -> tuple[NormalizedResponse | None, AttritionOutcome, int]:
        self._assert_runtime_identity(plan, messages)
        reusable = self._reusable_attempt(plan.trial_id, stage)
        if reusable:
            response = _response_from_record(reusable)
            return response, classify_transport(response), int(reusable["attempt_number"])
        existing = [row for row in self.api_attempts
                    if row.get("trial_id") == plan.trial_id and row.get("stage") == stage]
        if len(existing) >= self.parameters.retry_policy.max_attempts:
            response = _response_from_record(existing[-1])
            return response, classify_transport(response), int(existing[-1]["attempt_number"])
        config = self._configs[plan.config_id]
        client = ProviderAgnosticClient(config)
        request = client.build_request(
            messages=messages, temperature=self.parameters.temperature, max_tokens=max_tokens,
        )
        request = RequestSpec(
            base_url=request.base_url, model_id=request.model_id,
            api_key_env=request.api_key_env, body=request.body,
            metadata={**request.metadata, "stage": stage, "trial_id": plan.trial_id},
        )
        last_response: NormalizedResponse | None = None
        last_outcome = AttritionOutcome("infrastructure", "other_infrastructure", False)
        for attempt_number in range(len(existing) + 1, self.parameters.retry_policy.max_attempts + 1):
            limiter = self._limiter(plan.config_id, config)
            limiter.before_request(estimated_tokens=max_tokens)
            response = self._transport_for(plan.config_id).complete(request)
            outcome = classify_transport(response)
            record = self._attempt_record(plan, stage, attempt_number, request, response, outcome)
            self.storage.append("api_attempts.jsonl", record)
            self.api_attempts.append(record)
            self.hook(f"after_api_attempt_{'message' if stage == 'message' else 'final'}", record)
            last_response, last_outcome = response, outcome
            if outcome.category == "success":
                return response, outcome, attempt_number
            self._append_error(plan, stage, attempt_number, outcome, response)
            if outcome.quota_pause:
                self._pause_quota(plan, stage, outcome, response)
                return None, outcome, attempt_number
            if not outcome.retryable or attempt_number >= self.parameters.retry_policy.max_attempts:
                return response, outcome, attempt_number
            self._set_state(plan.trial_id, TrialState.INFRASTRUCTURE_PENDING_RETRY)
            delay = self.parameters.retry_policy.delay(
                failed_attempt_number=attempt_number,
                retry_after_seconds=response.retry_after_seconds,
                random_value=self.retry_random,
            )
            self.retry_sleeper(delay)
            target = TrialState.MESSAGE_PENDING if stage == "message" else TrialState.DECISION_PENDING
            self._set_state(plan.trial_id, target)
        return last_response, last_outcome, len(existing)

    def _attempt_record(
        self, plan: PlannedTrial, stage: str, attempt_number: int,
        request: RequestSpec, response: NormalizedResponse, outcome: AttritionOutcome,
    ) -> dict[str, Any]:
        assignment = self.assignments.get(plan.trial_id, {})
        original_partner_id = assignment.get("original_partner_id")
        final_partner_id = assignment.get("final_partner_id")
        if stage == "message":
            original_partner_id = f"{plan.trial_id}:B1"
        return sanitize_for_serialization({
            "attempt_id": "attempt:" + hashlib.sha256(
                f"{plan.trial_id}|{stage}|{attempt_number}".encode()
            ).hexdigest()[:20],
            "trial_id": plan.trial_id,
            "stage": stage,
            "attempt_number": attempt_number,
            "timestamp": self.timestamp(),
            "run_id": plan.run_id,
            "randomization_seed": plan.randomization_seed,
            "condition": assignment.get("condition") or plan.final_condition,
            "original_partner_id": original_partner_id,
            "final_partner_id": final_partner_id,
            "prompt_version": PROMPT_VERSION,
            "provider": plan.provider,
            "model": plan.model,
            "reasoning_mode": plan.reasoning_mode,
            "plan_phase": plan.plan_phase,
            "top_up_sequence": plan.top_up_sequence,
            "top_up_block_id": plan.top_up_block_id,
            "request": request.to_log_dict(),
            "response": response.to_dict(),
            "transport_category": outcome.category,
            "failure_reason": outcome.cause,
        })

    def _append_trial_result(
        self, plan: PlannedTrial, *, final_action: str | None,
        raw_final_response: str, completion_status: str,
        failure_category: str | None, failure_reason: str | None,
        attempt_number: int,
    ) -> None:
        if plan.trial_id in self.trials:
            return
        assignment = self.assignments.get(plan.trial_id)
        if assignment:
            condition = Condition(assignment["condition"])
            metadata = treatment_metadata(condition, plan.trial_id)
            original_partner_id = assignment["original_partner_id"]
            final_partner_id = assignment["final_partner_id"]
            switched = assignment["switched"]
        else:
            # A voluntary message-stage infrastructure failure precedes assignment.
            condition = None
            metadata = None
            original_partner_id = f"{plan.trial_id}:B1"
            final_partner_id = original_partner_id
            switched = False
        event = self.message_events.get(plan.trial_id, {})
        row = sanitize_for_serialization({
            "trial_id": plan.trial_id,
            "run_id": plan.run_id,
            "experiment_version": EXPERIMENT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "randomization_seed": plan.randomization_seed,
            "condition": condition.value if condition else None,
            "model": plan.model,
            "provider": plan.provider,
            "reasoning_mode": plan.reasoning_mode,
            "plan_phase": plan.plan_phase,
            "top_up_sequence": plan.top_up_sequence,
            "top_up_block_id": plan.top_up_block_id,
            "message_regime": metadata.message_regime.value if metadata else "voluntary",
            "assigned_expectation": metadata.assigned_expectation if metadata else False,
            "original_partner_id": original_partner_id,
            "final_partner_id": final_partner_id,
            "switched": switched,
            "own_promise_regime": metadata.own_promise_regime.value if metadata else "voluntary_opportunity",
            "final_action": final_action,
            "completion_status": completion_status,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "attempt_number": attempt_number,
            "raw_message": event.get("raw_message", ""),
            "raw_message_response": event.get("raw_response", ""),
            "message_recorded": bool(event),
            "message_event_id": event.get("message_event_id"),
            "raw_final_response": raw_final_response,
        })
        self.storage.append("trials.jsonl", row)
        self.trials[plan.trial_id] = row
        self.hook("after_trial_result", row)

    def _append_error(
        self, plan: PlannedTrial, stage: str, attempt_number: int,
        outcome: AttritionOutcome, response: NormalizedResponse,
    ) -> None:
        self.storage.append("errors.jsonl", sanitize_for_serialization({
            "trial_id": plan.trial_id, "stage": stage,
            "attempt_number": attempt_number, "timestamp": self.timestamp(),
            "provider": plan.provider, "model": plan.model,
            "category": outcome.category, "cause": outcome.cause,
            "retryable": outcome.retryable, "quota_pause": outcome.quota_pause,
            "http_status": response.http_status,
            "provider_request_id": response.provider_request_id,
            "error_metadata": response.error_metadata,
        }))

    def _pause_quota(
        self, plan: PlannedTrial, stage: str,
        outcome: AttritionOutcome, response: NormalizedResponse,
    ) -> None:
        self._set_state(plan.trial_id, TrialState.PAUSED_QUOTA)
        self.run_status = RunStatus.PAUSED_PROVIDER_QUOTA
        self.pause_detail = sanitize_for_serialization({
            "provider": plan.provider, "model": plan.model, "stage": stage,
            "timestamp": self.timestamp(),
            "completed_count": sum(
                row.get("completion_status") == "completed" for row in self.trials.values()
            ),
            "pending_count": sum(not is_terminal(state) for state in self.states.values()),
            "failure_reason": outcome.cause,
            "http_status": response.http_status,
            "provider_request_id": response.provider_request_id,
            "error_metadata": response.error_metadata,
        })
        self._record_pause_event()
        self._persist_state()

    def _pause_infrastructure_review(
        self,
        plan: PlannedTrial,
        stage: str,
        outcome: AttritionOutcome,
        response: NormalizedResponse,
        attempt_number: int,
    ) -> None:
        """Pause the batch after one trial exhausts its bounded infrastructure path."""
        self.run_status = RunStatus.PAUSED_INFRASTRUCTURE_REVIEW
        self.pause_detail = sanitize_for_serialization({
            "provider": plan.provider,
            "model": plan.model,
            "stage": stage,
            "trial_id": plan.trial_id,
            "timestamp": self.timestamp(),
            "failure_reason": outcome.cause,
            "attempt_number": attempt_number,
            "http_status": response.http_status,
            "provider_request_id": response.provider_request_id,
            "required_action": "manual infrastructure review before resume",
        })
        self._record_pause_event()
        self._persist_state()

    def _behavioral_parser_failure_counts(self) -> dict[str, dict[str, Any]]:
        counts = {
            config_id: {
                "provider": config.provider,
                "model": config.model_id,
                "reasoning_mode": config.reasoning_mode,
                "message": 0,
                "final_decision": 0,
                "total": 0,
            }
            for config_id, config in self._configs.items()
        }
        plans = {plan.trial_id: plan for plan in self.plans}
        for trial_id, event in self.message_events.items():
            plan = plans.get(trial_id)
            if plan and event.get("behavioral_failure_reason"):
                counts[plan.config_id]["message"] += 1
        for trial_id, trial in self.trials.items():
            plan = plans.get(trial_id)
            if (
                plan
                and trial.get("failure_category") == "behavioral"
                and int(trial.get("attempt_number") or 0) > 0
            ):
                counts[plan.config_id]["final_decision"] += 1
        for row in counts.values():
            row["total"] = row["message"] + row["final_decision"]
        return counts

    def _maybe_pause_behavioral_parser_review(
        self,
        plan: PlannedTrial,
        stage: str,
        outcome: AttritionOutcome,
        attempt_number: int,
    ) -> None:
        """Pause on the second durable parser failure; compatibility remains manual."""
        count = self._behavioral_parser_failure_counts()[plan.config_id]["total"]
        if count < BEHAVIORAL_PARSER_FAILURE_REVIEW_COUNT:
            return
        self.run_status = RunStatus.PAUSED_BEHAVIORAL_PARSER_REVIEW
        self.pause_detail = sanitize_for_serialization({
            "provider": plan.provider,
            "model": plan.model,
            "reasoning_mode": plan.reasoning_mode,
            "model_config_id": plan.config_id,
            "stage": stage,
            "trial_id": plan.trial_id,
            "timestamp": self.timestamp(),
            "failure_reason": outcome.cause,
            "attempt_number": attempt_number,
            "behavioral_parser_failure_count": count,
            "review_trigger_count": BEHAVIORAL_PARSER_FAILURE_REVIEW_COUNT,
            "required_action": (
                "manual model-configuration response-contract review before resume"
            ),
        })
        self._record_pause_event()
        self._persist_state()

    def _record_pause_event(self) -> None:
        detail = self.pause_detail or {}
        identity = "|".join(str(detail.get(key, "")) for key in (
            "trial_id", "stage", "attempt_number", "failure_reason",
            "behavioral_parser_failure_count", "provider_request_id",
        ))
        event = sanitize_for_serialization({
            "pause_event_id": "pause:" + hashlib.sha256(identity.encode()).hexdigest()[:20],
            "run_id": self.parameters.run_id,
            "run_status": self.run_status.value,
            "detail": detail,
        })
        existing = {row.get("pause_event_id") for row in self.storage.read_jsonl("pause_events.jsonl")}
        if event["pause_event_id"] not in existing:
            self.storage.append("pause_events.jsonl", event)

    def _set_state(self, trial_id: str, target: TrialState) -> None:
        current = self.states[trial_id]
        if current == target:
            return
        self.states[trial_id] = transition(current, target)
        self._persist_state()

    def _persist_state(self) -> None:
        self.storage.write_state({
            "execution_schema_version": EXECUTION_SCHEMA_VERSION,
            "run_id": self.parameters.run_id,
            "run_status": self.run_status.value,
            "updated_at": self.timestamp(),
            "pause_detail": self.pause_detail,
            "trials": {key: value.value for key, value in self.states.items()},
            "summary": self.technical_summary() if self.plans else {},
        })

    def _manifest(self) -> dict[str, Any]:
        normalized = self._normalized_parameters()
        snapshot = {
            "experiment_version": EXPERIMENT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_templates_hash(),
            "parameters": normalized,
        }
        git_commit, dirty = _git_metadata()
        return sanitize_for_serialization({
            "execution_schema_version": EXECUTION_SCHEMA_VERSION,
            "run_id": self.parameters.run_id,
            "dataset_role": self.parameters.dataset_role,
            "experiment_version": EXPERIMENT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_templates_hash(),
            "created_at": self.timestamp(),
            "randomization_seed": self.parameters.seed,
            "requested_completed_trials_per_final_condition": self.parameters.n_per_condition,
            "completion_policy": {
                "version": COMPLETION_POLICY_VERSION,
                "minimum_completed_binary_per_condition": self.parameters.n_per_condition,
                "top_up_enabled": self.parameters.top_up_enabled,
                "stopping_inputs": [
                    "completed", "behavioral_missing", "infrastructure_attrition",
                    "pending", "retryable", "paused",
                ],
                "prohibited_stopping_inputs": [
                    "action_value", "cooperation_rate", "effect", "p_value",
                    "confidence_interval", "message_content",
                ],
                "voluntary_top_up_unit": "randomized_balanced_block_of_2",
            },
            "model_configurations": [config.to_dict() for config in self.parameters.model_configs],
            "providers": sorted({config.provider for config in self.parameters.model_configs}),
            "reasoning_modes": sorted({config.reasoning_mode for config in self.parameters.model_configs}),
            "temperature": self.parameters.temperature,
            "temperature_override_authorized": (
                self.parameters.temperature_override_authorized
            ),
            "token_limits": {"message": self.parameters.message_max_tokens,
                             "final_decision": self.parameters.decision_max_tokens},
            "max_output_tokens": (
                self.parameters.decision_max_tokens
                if self.parameters.message_max_tokens == self.parameters.decision_max_tokens
                else None
            ),
            "output_cap_override_authorized": (
                self.parameters.output_cap_override_authorized
            ),
            "retry_policy": self.parameters.retry_policy.to_dict(),
            "pacing_rate_limit_policy": self.parameters.rate_limit_policy.to_dict(),
            "config_snapshot": snapshot,
            "config_hash": _canonical_hash(snapshot),
            "git_commit": git_commit,
            "working_tree_dirty": dirty,
            "cli_invocation": list(self.parameters.cli_invocation),
            "normalized_execution_parameters": normalized,
            "dry_run": self.parameters.dry_run,
            "top_up_enabled": self.parameters.top_up_enabled,
        })

    def _normalized_parameters(self) -> dict[str, Any]:
        return sanitize_for_serialization({
            "run_id": self.parameters.run_id,
            "dataset_role": self.parameters.dataset_role,
            "n_per_condition": self.parameters.n_per_condition,
            "seed": self.parameters.seed,
            "model_configurations": [config.to_dict() for config in self.parameters.model_configs],
            "temperature": self.parameters.temperature,
            "temperature_override_authorized": (
                self.parameters.temperature_override_authorized
            ),
            "message_max_tokens": self.parameters.message_max_tokens,
            "decision_max_tokens": self.parameters.decision_max_tokens,
            "output_cap_override_authorized": (
                self.parameters.output_cap_override_authorized
            ),
            "retry_policy": self.parameters.retry_policy.to_dict(),
            "rate_limit_policy": self.parameters.rate_limit_policy.to_dict(),
            "dry_run": self.parameters.dry_run,
            "top_up_enabled": self.parameters.top_up_enabled,
        })

    def _assert_runtime_identity(
        self, plan: PlannedTrial, messages: Sequence[dict[str, str]],
    ) -> None:
        """Stop before transport if frozen identity or prompt isolation changed mid-run."""
        if not self.manifest:
            raise RuntimeError("Run manifest is unavailable before provider request")
        if self.manifest.get("experiment_version") != EXPERIMENT_VERSION:
            raise RuntimeError("Catastrophic run-integrity failure: experiment version mismatch")
        if self.manifest.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError("Catastrophic run-integrity failure: prompt version mismatch")
        if self.manifest.get("prompt_hash") != prompt_templates_hash():
            raise RuntimeError("Catastrophic run-integrity failure: frozen prompt hash changed")
        if self.manifest.get("dataset_role") != self.parameters.dataset_role:
            raise RuntimeError("Catastrophic run-integrity failure: dataset role mismatch")
        config = self._configs.get(plan.config_id)
        if config is None or plan.model != config.model_id or plan.provider != config.provider:
            raise RuntimeError("Catastrophic run-integrity failure: model/config mismatch")
        prompt_text = json.dumps(list(messages), ensure_ascii=False).casefold()
        labels = [condition.value for condition in Condition]
        labels += [scientific_condition_label(condition) for condition in Condition]
        if any(label.casefold() in prompt_text for label in labels):
            raise RuntimeError("Catastrophic run-integrity failure: condition-label leakage")

    def _reusable_attempt(self, trial_id: str, stage: str) -> dict[str, Any] | None:
        rows = [row for row in self.api_attempts
                if row.get("trial_id") == trial_id and row.get("stage") == stage
                and row.get("transport_category") == "success"]
        return rows[-1] if rows else None

    def _transport_for(self, config_id: str) -> CompletionTransport:
        if isinstance(self.transport, Mapping):
            return self.transport[config_id]
        if self.transport is not None:
            return self.transport
        if self.parameters.dry_run:
            return self._default_dry_transport
        return OpenAICompatibleTransport()

    def _limiter(self, config_id: str, config: ModelConfig) -> RateLimiter:
        if config_id not in self._limiters:
            if self.rate_limiter_factory:
                limiter = self.rate_limiter_factory(config)
            elif self.parameters.dry_run:
                limiter = RateLimiter(RateLimitPolicy(
                    requests_per_minute=None, tokens_per_minute=None,
                    max_concurrency=1, minimum_interval_seconds=0.0,
                ))
            else:
                limiter = RateLimiter(self.parameters.rate_limit_policy)
            self._limiters[config_id] = limiter
        return self._limiters[config_id]

    def _plan(self, trial_id: str) -> PlannedTrial:
        return next(plan for plan in self.plans if plan.trial_id == trial_id)


def parameters_from_manifest(run_dir: Path) -> ExecutionParameters:
    manifest = RunStorage(run_dir).read_manifest()
    if manifest.get("execution_schema_version") != EXECUTION_SCHEMA_VERSION:
        raise ValueError("Refusing to resume a legacy or incompatible Promise Switch run")
    normalized = manifest.get("normalized_execution_parameters", {})
    configs = tuple(ModelConfig(**row) for row in normalized.get("model_configurations", []))
    return ExecutionParameters(
        run_id=normalized["run_id"], output_dir=run_dir.parent,
        n_per_condition=int(normalized["n_per_condition"]), seed=int(normalized["seed"]),
        model_configs=configs, dataset_role=normalized["dataset_role"],
        temperature=float(normalized["temperature"]),
        temperature_override_authorized=bool(
            normalized.get("temperature_override_authorized", False)
        ),
        message_max_tokens=int(normalized["message_max_tokens"]),
        decision_max_tokens=int(normalized["decision_max_tokens"]),
        output_cap_override_authorized=bool(
            normalized.get("output_cap_override_authorized", False)
        ),
        retry_policy=RetryPolicy(**normalized["retry_policy"]),
        rate_limit_policy=RateLimitPolicy(**normalized["rate_limit_policy"]),
        dry_run=bool(normalized["dry_run"]),
        top_up_enabled=bool(normalized.get("top_up_enabled", False)),
    )


def _last_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def _response_from_record(record: dict[str, Any]) -> NormalizedResponse:
    return NormalizedResponse(**record["response"])


def _allocation_seed(seed: int, config_id: str, group_id: str) -> int:
    return int(hashlib.sha256(
        f"{seed}|{config_id}|voluntary|{group_id}".encode()
    ).hexdigest()[:16], 16)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_metadata() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
