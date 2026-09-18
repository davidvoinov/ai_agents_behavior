"""Durable Promise Switch scientific orchestration over provider Batch jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..client import RequestSpec
from ..design import Condition
from ..schemas import AssignmentRecord, MessageEvent, NormalizedResponse
from .batch_transport import BatchCoordinator, BatchLifecycleState, BatchProvider
from .batch_workflow import build_decision_batch_requests, build_message_batch_requests
from .lock import RunWriterLock
from .planner import PlannedTrial
from .runner import ExecutionParameters, PromiseSwitchRunner
from .state import RunStatus, TrialState, is_terminal


class PromiseSwitchBatchRunner:
    """Explicit prepare/retrieve/ingest driver; never contacts a provider implicitly."""

    def __init__(
        self, parameters: ExecutionParameters, provider: BatchProvider,
        *, timestamp: Callable[[], str] | None = None,
    ) -> None:
        if parameters.execution_mode != "provider_batch":
            raise ValueError("PromiseSwitchBatchRunner requires provider_batch execution mode")
        if len(parameters.model_configs) != 1:
            raise ValueError("one provider Batch run must contain exactly one model configuration")
        if provider.name != parameters.model_configs[0].provider:
            raise ValueError("Batch provider does not match the model configuration")
        self.parameters = parameters
        self.provider = provider
        self.timestamp = timestamp or (lambda: datetime.now(timezone.utc).isoformat())
        self.coordinator = BatchCoordinator(
            parameters.run_dir, provider, timestamp=self.timestamp,
        )

    def initialize(self) -> dict[str, Any]:
        return PromiseSwitchRunner(self.parameters, timestamp=self.timestamp).initialize()

    def prepare_next_batch(self, *, batch_sequence: int) -> str | None:
        """Advance durable non-provider states and prepare exactly one next phase."""
        with RunWriterLock(self.parameters.run_dir, run_id=self.parameters.run_id):
            runner = self._loaded_runner()
            self._reject_unresolved_batches()
            if runner.run_status == RunStatus.CREATED:
                runner.run_status = RunStatus.RUNNING
                runner._persist_state()
            if runner.run_status in {
                RunStatus.PAUSED_PROVIDER_QUOTA,
                RunStatus.PAUSED_INFRASTRUCTURE_REVIEW,
                RunStatus.PAUSED_BEHAVIORAL_PARSER_REVIEW,
            }:
                raise RuntimeError(f"run is paused: {runner.run_status.value}")
            self._advance_non_provider_states(runner)
            if runner._append_required_message_recovery_trials():
                self._advance_non_provider_states(runner)
            pending = self._pending_requests(runner)
            if pending is None:
                if runner.plans and all(is_terminal(state) for state in runner.states.values()):
                    if self.parameters.top_up_enabled and not runner._completion_target_met():
                        if runner._append_required_top_ups():
                            self._advance_non_provider_states(runner)
                            pending = self._pending_requests(runner)
                    elif runner._completion_target_met():
                        runner.run_status = RunStatus.COMPLETED
                        runner._persist_state()
                if pending is None:
                    return None
            phase, requests = pending
        # Coordinator acquires the same run lock and performs the final global
        # uniqueness check immediately before its durable PREPARED journal.
        config = self.parameters.model_configs[0]
        return self.coordinator.prepare(
            run_id=self.parameters.run_id, model=config.model_id,
            reasoning_mode=config.reasoning_mode, batch_sequence=batch_sequence,
            batch_phase=phase, requests=requests,
        )

    def poll_and_retrieve(
        self, batch_key: str,
    ) -> dict[str, NormalizedResponse] | None:
        """Explicit provider operation; callers decide when network I/O is allowed."""
        return self.coordinator.poll_and_retrieve(batch_key)

    def resume_paused_run(self, *, approve_parser_review_class: str | None = None) -> None:
        """Explicitly clear an operator-reviewed pause; never called automatically."""
        with RunWriterLock(self.parameters.run_dir, run_id=self.parameters.run_id):
            runner = self._loaded_runner(
                approve_parser_review_class=approve_parser_review_class,
            )
            if runner.run_status not in {
                RunStatus.PAUSED_PROVIDER_QUOTA,
                RunStatus.PAUSED_INFRASTRUCTURE_REVIEW,
                RunStatus.PAUSED_BEHAVIORAL_PARSER_REVIEW,
            }:
                raise ValueError("run has no active pause to resume")
            runner.run_status = RunStatus.RUNNING
            runner.pause_detail = None
            runner._persist_state()

    def ingest_retrieved(
        self, batch_key: str, responses: Mapping[str, NormalizedResponse],
    ) -> dict[str, str]:
        """Durably ingest every known result, leaving missing IDs in reconciliation."""
        with RunWriterLock(self.parameters.run_dir, run_id=self.parameters.run_id):
            current = self.coordinator.latest(batch_key)
            if current is None or current.get("status") not in {
                BatchLifecycleState.RESULTS_RETRIEVED.value,
                BatchLifecycleState.INGESTED.value,
            }:
                raise ValueError("batch results are not ready for ingestion")
            mappings = self.coordinator.mappings(batch_key)
            mapping_by_id = {str(row["custom_id"]): row for row in mappings}
            if set(responses) - set(mapping_by_id):
                raise ValueError("result references an unexpected custom_id")
            runner = self._loaded_runner()
            runner.run_status = RunStatus.RUNNING
            plans = {plan.trial_id: plan for plan in runner.plans}
            outcomes: dict[str, str] = {}
            for custom_id, response in responses.items():
                row = mapping_by_id[custom_id]
                plan = plans.get(str(row["trial_id"]))
                if plan is None:
                    raise ValueError("batch mapping references an unknown trial")
                request_row = row["request"]
                request = RequestSpec(
                    base_url=str(request_row["base_url"]),
                    model_id=str(request_row["model_id"]),
                    api_key_env=str(request_row["api_key_env"]),
                    body=dict(request_row["body"]), metadata=dict(request_row["metadata"]),
                )
                outcomes[custom_id] = runner.ingest_batch_response(
                    plan=plan, stage=str(row["stage"]),
                    attempt_number=int(row["attempt_number"]), request=request,
                    response=response, batch_key=batch_key, custom_id=custom_id,
                    request_body_hash=str(row["request_body_hash"]),
                    provider_job_id=(str(current["provider_job_id"])
                                     if current.get("provider_job_id") else None),
                )
            # Reconstruct from disk, never from caller-supplied MessageEvent objects.
            runner._reconcile()
            if runner.run_status != RunStatus.PAUSED_BEHAVIORAL_PARSER_REVIEW:
                runner._allocate_ready_voluntary_pools()
            durable_events = {
                row["message_event_id"]: row
                for row in runner.storage.read_jsonl("message_events.jsonl")
            }
            for assignment in runner.storage.read_jsonl("assignments.jsonl"):
                if assignment.get("condition") in {
                    Condition.VOLUNTARY_PROMISE_NO_SWITCH.value,
                    Condition.VOLUNTARY_PROMISE_SWITCH.value,
                } and assignment.get("message_event_id") not in durable_events:
                    raise RuntimeError("voluntary assignment lacks a durable message event")
            runner._persist_state()
        self.coordinator.mark_ingested(batch_key)
        return outcomes

    def _loaded_runner(
        self, *, approve_parser_review_class: str | None = None,
    ) -> PromiseSwitchRunner:
        runner = PromiseSwitchRunner(self.parameters, timestamp=self.timestamp)
        runner._load_for_resume(approve_parser_review_class)
        runner._reconcile()
        state = runner.storage.read_state()
        if state and state.get("run_status"):
            runner.run_status = RunStatus(str(state["run_status"]))
            runner.pause_detail = state.get("pause_detail")
        return runner

    def _reject_unresolved_batches(self) -> None:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.coordinator.storage.read_jsonl("batch_jobs.jsonl"):
            latest[str(row["batch_key"])] = row
        blocking = {
            key: row["status"] for key, row in latest.items()
            if row.get("status") != BatchLifecycleState.INGESTED.value
            and not (
                row.get("status") == BatchLifecycleState.RECONCILIATION_REQUIRED.value
                and row.get("known_results_ingested") is True
                and row.get("missing_custom_ids")
            )
        }
        if blocking:
            raise RuntimeError(f"existing Batch lifecycle requires resolution: {blocking}")

    def _advance_non_provider_states(self, runner: PromiseSwitchRunner) -> None:
        for plan in runner.plans:
            if runner.states[plan.trial_id] == TrialState.PLANNED:
                if plan.planned_regime == "voluntary":
                    runner._set_state(plan.trial_id, TrialState.MESSAGE_PENDING)
                else:
                    runner._append_fixed_assignment(plan)
        runner._allocate_ready_voluntary_pools()
        for plan in runner.plans:
            if runner.states[plan.trial_id] == TrialState.ASSIGNED:
                event = runner.message_events.get(plan.trial_id)
                if event and event.get("behavioral_failure_reason"):
                    runner._append_trial_result(
                        plan, final_action=None, raw_final_response="",
                        completion_status="failed", failure_category="behavioral",
                        failure_reason=event["behavioral_failure_reason"], attempt_number=0,
                    )
                    runner._set_state(plan.trial_id, TrialState.BEHAVIORAL_FAILURE)
                else:
                    runner._set_state(plan.trial_id, TrialState.DECISION_PENDING)

    def _pending_requests(
        self, runner: PromiseSwitchRunner,
    ) -> tuple[str, list[tuple[RequestSpec, int]]] | None:
        config = self.parameters.model_configs[0]
        message_plans = [
            plan for plan in runner.plans
            if runner.states[plan.trial_id] == TrialState.MESSAGE_PENDING
            and self._attempt_is_preparable(runner, plan, "message")
        ]
        if message_plans:
            requests = build_message_batch_requests(
                message_plans, config=config,
                max_tokens=int(self.parameters.message_max_tokens),
                assignments=runner.assignments,
            )
            return "message", [
                (request, self._next_attempt(runner, plan, "message"))
                for plan, request in zip(message_plans, requests, strict=True)
            ]
        decision_plans = [
            plan for plan in runner.plans
            if runner.states[plan.trial_id] == TrialState.DECISION_PENDING
            and self._attempt_is_preparable(runner, plan, "final_decision")
        ]
        if not decision_plans:
            return None
        assignments = {
            trial_id: _assignment(row) for trial_id, row in runner.assignments.items()
        }
        events = {
            trial_id: MessageEvent(
                message_event_id=str(row["message_event_id"]), trial_id=trial_id,
                timestamp=str(row["timestamp"]), raw_message=str(row["raw_message"]),
                message_recorded=bool(row["message_recorded"]),
            )
            for trial_id, row in runner.message_events.items()
        }
        requests = build_decision_batch_requests(
            decision_plans, config=config,
            max_tokens=int(self.parameters.decision_max_tokens),
            assignments=assignments, message_events=events,
        )
        return "final_decision", [
            (request, self._next_attempt(runner, plan, "final_decision"))
            for plan, request in zip(decision_plans, requests, strict=True)
        ]

    def _attempt_is_preparable(
        self, runner: PromiseSwitchRunner, plan: PlannedTrial, stage: str,
    ) -> bool:
        attempt = self._next_attempt(runner, plan, stage)
        key = (plan.trial_id, stage, attempt)
        mapped = {
            (str(row.get("trial_id")), str(row.get("stage")), int(row.get("attempt_number", 0)))
            for row in self.coordinator.storage.read_jsonl("batch_request_mappings.jsonl")
        }
        return key not in mapped

    def _next_attempt(
        self, runner: PromiseSwitchRunner, plan: PlannedTrial, stage: str,
    ) -> int:
        prior = [
            int(row["attempt_number"]) for row in runner.api_attempts
            if row.get("trial_id") == plan.trial_id and row.get("stage") == stage
        ]
        attempt = max(prior, default=0) + 1
        if attempt > self.parameters.retry_policy.max_attempts:
            raise RuntimeError("refusing to create a fourth scientific attempt")
        return attempt


def _assignment(row: Mapping[str, Any]) -> AssignmentRecord:
    return AssignmentRecord(
        assignment_id=str(row["assignment_id"]), trial_id=str(row["trial_id"]),
        randomization_seed=int(row["randomization_seed"]),
        condition=Condition(str(row["condition"])),
        message_event_id=(str(row["message_event_id"])
                          if row.get("message_event_id") is not None else None),
        original_partner_id=str(row["original_partner_id"]),
        final_partner_id=str(row["final_partner_id"]), switched=bool(row["switched"]),
    )
