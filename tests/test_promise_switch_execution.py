from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from promise_switch.client import RequestSpec
from promise_switch.design import Condition
from promise_switch.analysis import VOLUNTARY_ITT_CONTRAST, binary_contrast
from promise_switch.artifacts import read_jsonl, validate_run_integrity
from promise_switch.execution.attrition import classify_transport
from promise_switch.execution.lock import RunLockError, RunWriterLock
from promise_switch.execution.planner import model_config_id, plan_trials
from promise_switch.execution.rate_limit import RateLimiter, RateLimitPolicy
from promise_switch.execution.retry import RetryPolicy
from promise_switch.execution.runner import (
    ExecutionParameters,
    PromiseSwitchRunner,
    parameters_from_manifest,
)
from promise_switch.execution.scheduler import balanced_pending_order
from promise_switch.execution.state import TrialState, transition
from promise_switch.execution.storage import FILES, RunStorage
from promise_switch.execution.transport import (
    normalize_openai_response,
    normalize_reasoning_content,
)
from promise_switch.parser import parse_decision
from promise_switch.providers import create_model_config
from promise_switch.randomization import allocate_voluntary_pool
from promise_switch.schemas import MessageEvent, NormalizedResponse
from promise_switch.version import RESPONSE_QC_VERSION, prompt_templates_hash
from scripts.audit_promise_switch_artifacts import audit as audit_run


def success_for(request: RequestSpec, suffix: str = "") -> NormalizedResponse:
    if request.metadata["stage"] == "message":
        raw = json.dumps({"message": "optional" + suffix})
    else:
        raw = json.dumps({"decision": "COOPERATE", "justification": "brief"})
    return NormalizedResponse(
        status="success", raw_response=raw, http_status=200,
        provider_request_id="req-safe", finish_reason="stop", latency_seconds=0.01,
        usage={"total_tokens": 5},
    )


class FakeTransport:
    def __init__(self, scripted=None, *, message_suffix=""):
        self.scripted = list(scripted or [])
        self.message_suffix = message_suffix
        self.calls: list[RequestSpec] = []

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.calls.append(request)
        if self.scripted:
            item = self.scripted.pop(0)
            return item(request) if callable(item) else item
        return success_for(request, self.message_suffix)


class FixedDecisionTransport(FakeTransport):
    def __init__(self, decision: str, justification: str = "test"):
        super().__init__()
        self.decision = decision
        self.justification = justification

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.calls.append(request)
        if request.metadata["stage"] == "message":
            raw = '{"message":"optional"}'
        else:
            raw = json.dumps({
                "decision": self.decision,
                "justification": self.justification,
            })
        return NormalizedResponse(
            status="success", raw_response=raw, http_status=200, finish_reason="stop",
        )


def request_condition(request: RequestSpec) -> Condition | None:
    if request.metadata.get("stage") != "final_decision":
        return None
    text = request.body["messages"][-1]["content"]
    if "Before assignment, you sent" in text:
        return (Condition.VOLUNTARY_PROMISE_SWITCH if "final partner is now" in text
                else Condition.VOLUNTARY_PROMISE_NO_SWITCH)
    if "Before assignment, you made this promise" in text:
        return (Condition.INDUCED_PROMISE_SWITCH if "final partner is now" in text
                else Condition.INDUCED_PROMISE_NO_SWITCH)
    if "You did not make a promise." in text:
        return Condition.EXPECTATION_WITHOUT_OWN_PROMISE
    if "No message was sent before the final decision." in text:
        return Condition.NO_COMMUNICATION
    raise AssertionError("unrecognized frozen condition prompt")


class OneFailureTransport(FakeTransport):
    def __init__(self, condition: Condition, *, action: str = "COOPERATE", infrastructure=False):
        super().__init__()
        self.condition = condition
        self.action = action
        self.infrastructure = infrastructure
        self.target_trial: str | None = None
        self.failed = False

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.calls.append(request)
        if request.metadata["stage"] == "message":
            return NormalizedResponse(status="success", raw_response='{"message":"optional"}',
                                      http_status=200, finish_reason="stop")
        if request_condition(request) == self.condition and not self.failed:
            if self.target_trial is None:
                self.target_trial = str(request.metadata["trial_id"])
            if request.metadata["trial_id"] == self.target_trial:
                if self.infrastructure:
                    attempts = sum(call.metadata.get("trial_id") == self.target_trial
                                   for call in self.calls)
                    if attempts <= 3:
                        if attempts == 3:
                            self.failed = True
                        return NormalizedResponse(status="timeout", raw_response="")
                else:
                    self.failed = True
                    return NormalizedResponse(status="success", raw_response="not-json",
                                              http_status=200, finish_reason="stop")
        return NormalizedResponse(
            status="success",
            raw_response=json.dumps({"decision": self.action, "justification": "brief"}),
            http_status=200, finish_reason="stop",
        )


class OneMessageInfrastructureFailureTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.target_trial: str | None = None

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.calls.append(request)
        if request.metadata["stage"] == "message":
            if self.target_trial is None:
                self.target_trial = str(request.metadata["trial_id"])
            if request.metadata["trial_id"] == self.target_trial:
                count = sum(call.metadata.get("trial_id") == self.target_trial for call in self.calls)
                if count <= 3:
                    return NormalizedResponse(status="timeout", raw_response="")
            return NormalizedResponse(status="success", raw_response='{"message":"optional"}',
                                      http_status=200, finish_reason="stop")
        return success_for(request)


class SeveralBehavioralFailuresTransport(FakeTransport):
    def __init__(self, condition: Condition, count: int):
        super().__init__()
        self.condition = condition
        self.count = count
        self.failed_trials: set[str] = set()

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.calls.append(request)
        if request.metadata["stage"] == "message":
            return NormalizedResponse(status="success", raw_response='{"message":"optional"}',
                                      http_status=200, finish_reason="stop")
        trial_id = str(request.metadata["trial_id"])
        if (request_condition(request) == self.condition
                and len(self.failed_trials) < self.count
                and trial_id not in self.failed_trials):
            self.failed_trials.add(trial_id)
            return NormalizedResponse(status="success", raw_response="not-json",
                                      http_status=200, finish_reason="stop")
        return success_for(request)


class OneShotCrash:
    def __init__(self, event: str, predicate=lambda _payload: True):
        self.event = event
        self.predicate = predicate
        self.fired = False

    def __call__(self, event, payload):
        if not self.fired and event == self.event and self.predicate(payload):
            self.fired = True
            raise RuntimeError("simulated crash")


class ExecutionTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.config = create_model_config(
            provider="nebius", model_alias="qwen3_32b", reasoning_mode="off"
        )

    def tearDown(self):
        self.temp.cleanup()

    def params(self, run_id="run", n=1, seed=17, **kwargs):
        dry_run = bool(kwargs.get("dry_run", False))
        return ExecutionParameters(
            run_id=run_id, output_dir=self.output, n_per_condition=n, seed=seed,
            model_configs=kwargs.pop("model_configs", (self.config,)),
            dataset_role=kwargs.pop(
                "dataset_role", "synthetic" if dry_run else "technical",
            ),
            retry_policy=kwargs.pop("retry_policy", RetryPolicy(
                max_retries=2, base_delay_seconds=0, max_delay_seconds=0,
                jitter_fraction=0,
            )),
            rate_limit_policy=kwargs.pop("rate_limit_policy", RateLimitPolicy(
                requests_per_minute=None, tokens_per_minute=None,
                max_concurrency=1, minimum_interval_seconds=0,
            )),
            **kwargs,
        )

    def runner(self, params, transport=None, **kwargs):
        return PromiseSwitchRunner(
            params, transport=transport or FakeTransport(),
            retry_sleeper=kwargs.pop("retry_sleeper", lambda _seconds: None),
            **kwargs,
        )

    def rows(self, params, name):
        return RunStorage(params.run_dir).read_jsonl(name)


class HappyPathTests(ExecutionTestCase):
    def test_initialize_persists_manifest_and_plan_without_transport(self):
        params = self.params(run_id="initialize-only")
        transport = FakeTransport()
        summary = self.runner(params, transport).initialize()
        self.assertEqual(summary["run_status"], "CREATED")
        self.assertEqual(summary["initial_planned_trials"], 6)
        self.assertEqual(transport.calls, [])
        manifest = RunStorage(params.run_dir).read_manifest()
        self.assertEqual(manifest["dataset_role"], "technical")
        self.assertEqual(len(read_jsonl(params.run_dir / "planned_trials.jsonl")), 6)
        validate_run_integrity(params.run_dir)

    def test_01_complete_execution_without_api(self):
        params = self.params(dry_run=True, n=2)
        summary = PromiseSwitchRunner(params).run()
        self.assertEqual(summary["run_status"], "COMPLETED")
        self.assertEqual(summary["completed_final_actions"], 12)
        self.assertEqual(len(self.rows(params, "api_attempts.jsonl")), 16)

    def test_02_manifest_and_required_files_created_before_calls(self):
        params = self.params()
        seen = []

        class CheckingTransport(FakeTransport):
            def complete(inner, request):
                seen.append((params.run_dir / "manifest.json").exists())
                return super().complete(request)

        self.runner(params, CheckingTransport()).run()
        self.assertTrue(all(seen))
        for name in ("manifest.json", "planned_trials.jsonl", "message_events.jsonl",
                     "assignments.jsonl", "api_attempts.jsonl", "trials.jsonl",
                     "errors.jsonl", "run_state.json"):
            self.assertTrue((params.run_dir / name).exists())

    def test_03_manifest_is_immutable(self):
        params = self.params()
        self.runner(params).run()
        storage = RunStorage(params.run_dir)
        with self.assertRaises(FileExistsError):
            storage.create_manifest({"replacement": True})

    def test_04_exact_voluntary_allocation(self):
        params = self.params(n=5)
        self.runner(params).run()
        counts = Counter(row["condition"] for row in self.rows(params, "assignments.jsonl"))
        self.assertEqual(counts[Condition.VOLUNTARY_PROMISE_NO_SWITCH.value], 5)
        self.assertEqual(counts[Condition.VOLUNTARY_PROMISE_SWITCH.value], 5)

    def test_05_response_content_does_not_change_assignment(self):
        first = self.params(run_id="same-science", n=3)
        self.runner(first, FakeTransport(message_suffix="-cooperate promise")).run()
        first_assignments = [(r["trial_id"], r["condition"])
                             for r in self.rows(first, "assignments.jsonl")
                             if r["message_event_id"]]
        other_output = Path(tempfile.mkdtemp(dir=self.temp.name))
        second = ExecutionParameters(**{**first.__dict__, "output_dir": other_output})
        self.runner(second, FakeTransport(message_suffix="-no promise")).run()
        second_assignments = [(r["trial_id"], r["condition"])
                              for r in RunStorage(second.run_dir).read_jsonl("assignments.jsonl")
                              if r["message_event_id"]]
        self.assertEqual(first_assignments, second_assignments)

    def test_06_fixed_conditions_are_interleaved(self):
        plans = plan_trials(run_id="x", seed=1, n_per_condition=4,
                            model_configs=(self.config,))
        fixed = [p.planned_regime for p in plans if p.planned_regime != "voluntary"]
        self.assertEqual(len(set(fixed[:4])), 4)
        self.assertEqual(fixed[:4], fixed[4:8])

    def test_07_balanced_scheduler_round_robins_cells(self):
        plans = plan_trials(run_id="x", seed=1, n_per_condition=3,
                            model_configs=(self.config,))
        states = {p.trial_id: TrialState.PLANNED for p in plans}
        conditions = {p.trial_id: p.final_condition for p in plans}
        ordered = balanced_pending_order(plans, states, conditions)
        lookup = {p.trial_id: p.planned_regime for p in plans}
        first_five = [lookup[trial_id] for trial_id in ordered[:5]]
        self.assertEqual(len(set(first_five)), 5)

    def test_08_illegal_voluntary_direct_assignment_rejected(self):
        with self.assertRaises(ValueError):
            transition(TrialState.MESSAGE_PENDING, TrialState.ASSIGNED)

    def test_08b_final_actions_are_recorded_exactly_and_linked_to_their_trials(self):
        scenarios = (
            ("COOPERATE", "test"),
            ("DEFECT", "test"),
            ("COOPERATE", "Defecting would give me more points."),
            ("DEFECT", "Cooperating would benefit both participants."),
        )
        for index, (decision, justification) in enumerate(scenarios):
            with self.subTest(decision=decision, justification=justification):
                params = self.params(run_id=f"action-recording-{index}")
                transport = FixedDecisionTransport(decision, justification)
                summary = self.runner(params, transport).run()
                self.assertEqual(summary["run_status"], "COMPLETED")
                trials = self.rows(params, "trials.jsonl")
                attempts = {
                    row["trial_id"]: row for row in self.rows(params, "api_attempts.jsonl")
                    if row["stage"] == "final_decision"
                }
                assignments = {
                    row["trial_id"]: row for row in self.rows(params, "assignments.jsonl")
                }
                self.assertEqual(len(trials), 6)
                self.assertEqual(set(attempts), {row["trial_id"] for row in trials})
                for trial in trials:
                    attempt = attempts[trial["trial_id"]]
                    assignment = assignments[trial["trial_id"]]
                    parsed = parse_decision(attempt["response"]["raw_response"])
                    self.assertTrue(parsed.valid)
                    self.assertEqual(parsed.decision, decision)
                    self.assertEqual(trial["final_action"], decision)
                    self.assertEqual(trial["completion_status"], "completed")
                    self.assertEqual(trial["raw_final_response"], parsed.raw_response)
                    self.assertEqual(attempt["trial_id"], trial["trial_id"])
                    self.assertEqual(
                        attempt["request"]["metadata"]["trial_id"], trial["trial_id"],
                    )
                    self.assertEqual(trial["condition"], assignment["condition"])
                    self.assertEqual(attempt["condition"], assignment["condition"])
                    self.assertEqual(
                        (trial["original_partner_id"], trial["final_partner_id"]),
                        (assignment["original_partner_id"], assignment["final_partner_id"]),
                    )
                    self.assertEqual(
                        (attempt["original_partner_id"], attempt["final_partner_id"]),
                        (assignment["original_partner_id"], assignment["final_partner_id"]),
                    )
                validate_run_integrity(params.run_dir, require_completed=True)

    def test_08c_parser_invalid_outputs_never_become_completed_actions(self):
        invalid_responses = (
            '```json\n{"decision":"COOPERATE","justification":"test"}\n```',
            "not json",
            "Decision: COOPERATE and DEFECT",
            '{"decision":"COOPERATE"}',
            '{"action":"DEFECT","justification":"test"}',
        )

        class FirstFinalInvalidTransport(FakeTransport):
            def __init__(inner, raw):
                super().__init__()
                inner.raw = raw
                inner.failed = False

            def complete(inner, request):
                inner.calls.append(request)
                if request.metadata["stage"] == "message":
                    return NormalizedResponse(
                        status="success", raw_response='{"message":"optional"}',
                        http_status=200, finish_reason="stop",
                    )
                if not inner.failed:
                    inner.failed = True
                    return NormalizedResponse(
                        status="success", raw_response=inner.raw,
                        http_status=200, finish_reason="stop",
                    )
                return success_for(request)

        for index, raw in enumerate(invalid_responses):
            with self.subTest(raw=raw):
                params = self.params(run_id=f"invalid-action-{index}")
                summary = self.runner(params, FirstFinalInvalidTransport(raw)).run()
                self.assertEqual(summary["run_status"], "COMPLETED")
                failed = [
                    row for row in self.rows(params, "trials.jsonl")
                    if row["raw_final_response"] == raw
                ]
                self.assertEqual(len(failed), 1)
                self.assertFalse(parse_decision(raw).valid)
                self.assertEqual(failed[0]["completion_status"], "failed")
                self.assertIsNone(failed[0]["final_action"])
                validate_run_integrity(params.run_dir, require_completed=True)


class CrashResumeTests(ExecutionTestCase):
    def _crash_and_resume(self, event, predicate=lambda _row: True):
        params = self.params(n=2)
        transport = FakeTransport()
        with self.assertRaises(RuntimeError):
            self.runner(params, transport, hook=OneShotCrash(event, predicate)).run()
        calls_at_crash = len(transport.calls)
        summary = self.runner(params, transport).run(resume=True)
        return params, transport, calls_at_crash, summary

    def test_09_crash_after_message_response_reuses_response(self):
        params, transport, calls, summary = self._crash_and_resume("after_api_attempt_message")
        self.assertEqual(summary["run_status"], "COMPLETED")
        self.assertEqual(len(transport.calls), 16)
        self.assertEqual(len(self.rows(params, "message_events.jsonl")), 4)
        self.assertEqual(calls, 1)

    def test_10_crash_after_durable_message_before_assignment(self):
        params, transport, _, summary = self._crash_and_resume("after_message_event")
        self.assertEqual(summary["run_status"], "COMPLETED")
        message_trials = [r["trial_id"] for r in self.rows(params, "message_events.jsonl")]
        self.assertEqual(len(message_trials), len(set(message_trials)))
        self.assertEqual(len(transport.calls), 16)

    def test_11_crash_after_voluntary_assignment_before_final(self):
        predicate = lambda row: row["condition"].startswith("voluntary_")
        params, transport, _, summary = self._crash_and_resume("after_assignment", predicate)
        self.assertEqual(summary["run_status"], "COMPLETED")
        assignments = self.rows(params, "assignments.jsonl")
        pairs = [(r["trial_id"], r["condition"]) for r in assignments]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(transport.calls), 16)

    def test_12_crash_after_final_response_reuses_response(self):
        params, transport, _, summary = self._crash_and_resume("after_api_attempt_final")
        self.assertEqual(summary["run_status"], "COMPLETED")
        self.assertEqual(len(transport.calls), 16)
        self.assertEqual(len(self.rows(params, "trials.jsonl")), 12)

    def test_13_resume_does_not_repeat_completed_trial(self):
        params = self.params(n=2)
        transport = FakeTransport()
        self.runner(params, transport).run()
        count = len(transport.calls)
        self.runner(params, transport).run(resume=True)
        self.assertEqual(len(transport.calls), count)
        self.assertEqual(len(self.rows(params, "trials.jsonl")), 12)

    def test_14_crash_resume_keeps_assignments_same_as_uninterrupted(self):
        params, _, _, _ = self._crash_and_resume("after_message_event")
        got = sorted((r["trial_id"], r["condition"])
                     for r in self.rows(params, "assignments.jsonl"))
        alternate = Path(tempfile.mkdtemp(dir=self.temp.name))
        uninterrupted = ExecutionParameters(**{**params.__dict__, "output_dir": alternate})
        self.runner(uninterrupted).run()
        expected = sorted((r["trial_id"], r["condition"])
                          for r in RunStorage(uninterrupted.run_dir).read_jsonl("assignments.jsonl"))
        self.assertEqual(got, expected)


class AttritionRetryTests(ExecutionTestCase):
    def test_15_behavioral_failure_not_retried(self):
        bad = NormalizedResponse(status="success", raw_response="not-json", http_status=200)
        transport = FakeTransport([bad])
        params = self.params()
        summary = self.runner(params, transport).run()
        first = self.rows(params, "trials.jsonl")[0]
        self.assertEqual(first["failure_category"], "behavioral")
        attempts = [r for r in self.rows(params, "api_attempts.jsonl")
                    if r["trial_id"] == first["trial_id"]]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(summary["behavioral_parser_failure_review_required"])
        self.assertFalse(summary["paused_behavioral_parser_review"])

    def test_15b_repeated_parser_failures_pause_for_manual_config_review(self):
        class TwoInvalidDecisions(FakeTransport):
            def __init__(inner):
                super().__init__()
                inner.invalid_decisions = 0

            def complete(inner, request):
                inner.calls.append(request)
                if (
                    request.metadata["stage"] == "final_decision"
                    and inner.invalid_decisions < 2
                ):
                    inner.invalid_decisions += 1
                    return NormalizedResponse(
                        status="success", raw_response="not-json", http_status=200,
                        finish_reason="stop",
                    )
                return success_for(request)

        params = self.params()
        transport = TwoInvalidDecisions()
        summary = self.runner(params, transport).run()
        self.assertEqual(summary["run_status"], "PAUSED_BEHAVIORAL_PARSER_REVIEW")
        self.assertTrue(summary["paused_behavioral_parser_review"])
        self.assertTrue(summary["behavioral_parser_failure_review_required"])
        self.assertEqual(summary["pause_detail"]["behavioral_parser_failure_count"], 2)
        failure_class = summary["pause_detail"]["parser_failure_class"]
        self.assertIn("manual", summary["pause_detail"]["required_action"])
        config_counts = next(iter(
            summary["behavioral_parser_failures_by_model_configuration"].values()
        ))
        self.assertEqual(config_counts["message"], 0)
        self.assertEqual(config_counts["final_decision"], 2)
        failures = [
            row for row in self.rows(params, "trials.jsonl")
            if row["failure_category"] == "behavioral"
        ]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(row["failure_reason"] == "invalid_output_format" for row in failures))
        failed_attempts = [
            row for row in self.rows(params, "api_attempts.jsonl")
            if row["trial_id"] in {failure["trial_id"] for failure in failures}
        ]
        self.assertEqual(len(failed_attempts), 2)
        with self.assertRaisesRegex(ValueError, "approve-parser-review-class"):
            self.runner(params, transport).run(resume=True)
        resumed = self.runner(params, transport).run(
            resume=True, approve_parser_review_class=failure_class,
        )
        self.assertEqual(resumed["run_status"], "COMPLETED")
        self.assertEqual(resumed["behavioral_failures"], 2)
        self.assertFalse(resumed["behavioral_parser_failure_review_required"])
        reviews = self.rows(params, "parser_review_events.jsonl")
        self.assertEqual([row["parser_failure_class"] for row in reviews], [failure_class])

    def test_15c_reviewed_class_survives_restart_but_different_class_still_pauses(self):
        missing = '{"decision":"DEFECT","justification":"brief"'
        missing_other_words = '{"decision":"DEFECT","justification":"different"'
        trailing_comma = '{"decision":"DEFECT","justification":"brief",}'

        class ContractFailures(FakeTransport):
            def __init__(inner):
                super().__init__()
                inner.failures = iter((missing, missing_other_words, missing, trailing_comma))

            def complete(inner, request):
                inner.calls.append(request)
                if request.metadata["stage"] == "final_decision":
                    try:
                        raw = next(inner.failures)
                    except StopIteration:
                        return success_for(request)
                    return NormalizedResponse(
                        status="success", raw_response=raw, http_status=200,
                        finish_reason="stop",
                    )
                return success_for(request)

        params = self.params(n=2)
        transport = ContractFailures()
        first = self.runner(params, transport).run()
        class_a = first["pause_detail"]["parser_failure_class"]
        self.assertEqual(class_a, "final_decision:missing_closing_delimiter")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.runner(params, transport).run(
                resume=True, approve_parser_review_class="final_decision:x:y",
            )
        second = self.runner(params, transport).run(
            resume=True, approve_parser_review_class=class_a,
        )
        self.assertEqual(second["run_status"], "PAUSED_BEHAVIORAL_PARSER_REVIEW")
        class_b = second["pause_detail"]["parser_failure_class"]
        self.assertEqual(class_b, "final_decision:trailing_comma")
        self.assertNotEqual(class_a, class_b)
        self.assertEqual(len(self.rows(params, "parser_review_events.jsonl")), 1)
        final = self.runner(params, transport).run(
            resume=True, approve_parser_review_class=class_b,
        )
        self.assertEqual(final["run_status"], "COMPLETED")
        failures = [
            row for row in self.rows(params, "trials.jsonl")
            if row.get("failure_category") == "behavioral"
        ]
        self.assertEqual(len(failures), 4)
        self.assertEqual(
            [row["raw_final_response"] for row in failures],
            [missing, missing_other_words, missing, trailing_comma],
        )
        failed_ids = {row["trial_id"] for row in failures}
        attempts = [
            row for row in self.rows(params, "api_attempts.jsonl")
            if row["trial_id"] in failed_ids
        ]
        self.assertEqual(len(attempts), 4)
        self.assertTrue(all(row["attempt_number"] == 1 for row in attempts))
        validate_run_integrity(params.run_dir, require_completed=True)

    def test_15d_residual_malformed_json_approval_is_occurrence_scoped(self):
        residuals = iter((
            '{"decision" "DEFECT"}',
            '{"decision" "COOPERATE"}',
            '{"justification" "brief"}',
        ))

        class ResidualFailures(FakeTransport):
            def complete(inner, request):
                inner.calls.append(request)
                if request.metadata["stage"] == "final_decision":
                    try:
                        raw = next(residuals)
                    except StopIteration:
                        return success_for(request)
                    return NormalizedResponse(
                        status="success", raw_response=raw, http_status=200,
                        finish_reason="stop",
                    )
                return success_for(request)

        params = self.params(n=2)
        transport = ResidualFailures()
        first = self.runner(params, transport).run()
        failure_class = first["pause_detail"]["parser_failure_class"]
        self.assertEqual(failure_class, "final_decision:other_malformed_json")
        second = self.runner(params, transport).run(
            resume=True, approve_parser_review_class=failure_class,
        )
        self.assertEqual(second["run_status"], "PAUSED_BEHAVIORAL_PARSER_REVIEW")
        self.assertEqual(second["pause_detail"]["parser_failure_class"], failure_class)
        final = self.runner(params, transport).run(
            resume=True, approve_parser_review_class=failure_class,
        )
        self.assertEqual(final["run_status"], "COMPLETED")
        reviews = self.rows(params, "parser_review_events.jsonl")
        self.assertEqual(len(reviews), 2)
        self.assertTrue(all(row["review_scope"] == "occurrence" for row in reviews))
        self.assertEqual(next(iter(final["reviewed_parser_failure_classes"].values())), [])
        validate_run_integrity(params.run_dir, require_completed=True)

    def test_15e_parser_class_approval_does_not_cross_runs_or_configurations(self):
        class TwoMissingDelimiters(FakeTransport):
            def __init__(inner):
                super().__init__()
                inner.remaining = 2

            def complete(inner, request):
                inner.calls.append(request)
                if request.metadata["stage"] == "final_decision" and inner.remaining:
                    inner.remaining -= 1
                    return NormalizedResponse(
                        status="success",
                        raw_response='{"decision":"DEFECT","justification":"brief"',
                        http_status=200, finish_reason="stop",
                    )
                return success_for(request)

        first_params = self.params(run_id="approval-run-a")
        first_transport = TwoMissingDelimiters()
        first = self.runner(first_params, first_transport).run()
        failure_class = first["pause_detail"]["parser_failure_class"]
        self.runner(first_params, first_transport).run(
            resume=True, approve_parser_review_class=failure_class,
        )
        second_params = self.params(run_id="approval-run-b")
        second = self.runner(second_params, TwoMissingDelimiters()).run()
        self.assertEqual(second["run_status"], "PAUSED_BEHAVIORAL_PARSER_REVIEW")
        self.assertEqual(second["pause_detail"]["parser_failure_class"], failure_class)

        other_config = create_model_config(
            provider="nebius", model_alias="qwen3_32b", reasoning_mode="on",
        )
        multi_params = self.params(
            run_id="approval-configs", model_configs=(self.config, other_config),
        )
        transports = {
            model_config_id(self.config): TwoMissingDelimiters(),
            model_config_id(other_config): TwoMissingDelimiters(),
        }
        first_config_pause = self.runner(multi_params, transports).run()
        approved_config = first_config_pause["pause_detail"]["model_config_id"]
        second_config_pause = self.runner(multi_params, transports).run(
            resume=True,
            approve_parser_review_class=first_config_pause["pause_detail"]["parser_failure_class"],
        )
        self.assertEqual(
            second_config_pause["run_status"], "PAUSED_BEHAVIORAL_PARSER_REVIEW",
        )
        self.assertNotEqual(
            second_config_pause["pause_detail"]["model_config_id"], approved_config,
        )

    def test_16_transient_failure_retries_and_attempt_numbers(self):
        rate = NormalizedResponse(status="error", raw_response="slow", http_status=429)
        transport = FakeTransport([rate, rate])
        params = self.params()
        self.runner(params, transport).run()
        attempts = self.rows(params, "api_attempts.jsonl")[:3]
        self.assertEqual([r["attempt_number"] for r in attempts], [1, 2, 3])
        self.assertEqual(attempts[0]["failure_reason"], "rate_limit_transient")

    def test_17_retry_after_is_respected(self):
        rate = NormalizedResponse(
            status="error", raw_response="slow", http_status=429, retry_after_seconds=7.0
        )
        sleeps = []
        params = self.params(retry_policy=RetryPolicy(
            max_retries=2, base_delay_seconds=1, max_delay_seconds=10, jitter_fraction=0
        ))
        self.runner(params, FakeTransport([rate]), retry_sleeper=sleeps.append).run()
        self.assertEqual(sleeps[0], 7.0)

    def test_18_exponential_backoff_policy(self):
        policy = RetryPolicy(base_delay_seconds=2, max_delay_seconds=99, jitter_fraction=0)
        self.assertEqual(policy.delay(failed_attempt_number=1), 2)
        self.assertEqual(policy.delay(failed_attempt_number=2), 4)
        self.assertEqual(policy.delay(failed_attempt_number=3), 8)

    def test_19_malformed_model_json_is_behavioral(self):
        bad = NormalizedResponse(status="success", raw_response="{", http_status=200)
        params = self.params()
        self.runner(params, FakeTransport([bad])).run()
        self.assertEqual(self.rows(params, "trials.jsonl")[0]["failure_category"], "behavioral")

    def test_20_malformed_api_envelope_is_infrastructure(self):
        bad = NormalizedResponse(status="malformed_api_response", raw_response="{}", http_status=200)
        params = self.params()
        self.runner(params, FakeTransport([bad, bad, bad])).run()
        first = self.rows(params, "trials.jsonl")[0]
        self.assertEqual(first["failure_category"], "infrastructure")
        self.assertEqual(first["failure_reason"], "malformed_api_response")

    def test_21_token_truncation_is_retryable_infrastructure(self):
        truncated = NormalizedResponse(
            status="success", raw_response="{", http_status=200, finish_reason="length"
        )
        outcome = classify_transport(truncated)
        self.assertEqual(outcome.cause, "token_truncation")
        self.assertTrue(outcome.retryable)
        params = self.params()
        transport = FakeTransport([truncated])
        self.runner(params, transport).run()
        self.assertGreater(len(transport.calls), 8)

    def test_22_quota_pause_and_resume(self):
        quota = NormalizedResponse(
            status="error", raw_response='{"error":"quota exhausted"}', http_status=429
        )
        params = self.params()
        transport = FakeTransport([quota])
        paused = self.runner(params, transport).run()
        self.assertEqual(paused["run_status"], "PAUSED_PROVIDER_QUOTA")
        state = RunStorage(params.run_dir).read_state()
        self.assertEqual(state["run_status"], "PAUSED_PROVIDER_QUOTA")
        finished = self.runner(params, transport).run(resume=True)
        self.assertEqual(finished["run_status"], "COMPLETED")

    def test_22b_exhausted_infrastructure_pauses_batch_for_review(self):
        unavailable = NormalizedResponse(
            status="error", raw_response="model unavailable", http_status=404,
        )
        params = self.params()
        transport = FakeTransport([unavailable])
        paused = self.runner(params, transport).run()
        self.assertEqual(paused["run_status"], "PAUSED_INFRASTRUCTURE_REVIEW")
        self.assertTrue(paused["paused_infrastructure_review"])
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(paused["pause_detail"]["failure_reason"], "model_unavailable")


class ResumeSafetyTests(ExecutionTestCase):
    def test_23_incompatible_resume_rejected(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        changed = ExecutionParameters(**{**params.__dict__, "seed": params.seed + 1})
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PromiseSwitchRunner(changed).run(resume=True)

    def test_24_changed_prompt_hash_rejected(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        with patch("promise_switch.execution.runner.prompt_templates_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "prompt hash"):
                PromiseSwitchRunner(params).run(resume=True)

    def test_25_legacy_run_rejected(self):
        params = self.params()
        params.run_dir.mkdir()
        (params.run_dir / "manifest.json").write_text('{"run_id":"run"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "legacy"):
            PromiseSwitchRunner(params).run(resume=True)

    def test_26_manifest_round_trip_parameters(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        restored = parameters_from_manifest(params.run_dir)
        self.assertEqual(restored.seed, params.seed)
        self.assertEqual(restored.model_configs, params.model_configs)

    def test_27_api_secret_absent_from_all_outputs(self):
        secret = "super-secret-nebius-value-123"
        params = self.params()
        with patch.dict(os.environ, {"NEBIUS_API_KEY": secret}):
            self.runner(params).run()
        for path in params.run_dir.iterdir():
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))

    def test_28_prompt_hash_logged(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        manifest = RunStorage(params.run_dir).read_manifest()
        self.assertEqual(manifest["prompt_hash"], prompt_templates_hash())
        self.assertEqual(manifest["dataset_role"], "synthetic")
        self.assertEqual(manifest["response_qc_version"], RESPONSE_QC_VERSION)
        self.assertEqual(RESPONSE_QC_VERSION, "promise_switch_response_qc_v3")


class RateLimitAndProviderTests(ExecutionTestCase):
    def test_29_rate_limiter_paces_proactively(self):
        current = [0.0]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            current[0] += seconds

        limiter = RateLimiter(
            RateLimitPolicy(requests_per_minute=None, minimum_interval_seconds=2),
            clock=lambda: current[0], sleeper=sleep,
        )
        limiter.before_request()
        limiter.before_request()
        self.assertEqual(sleeps, [2.0])

    def test_30_nebius_config_requires_only_env_name(self):
        self.assertEqual(self.config.api_key_env, "NEBIUS_API_KEY")
        self.assertIn("enable_thinking", self.config.provider_request_kwargs["chat_template_kwargs"])
        serialized = json.dumps(self.config.to_dict())
        self.assertNotIn("Bearer", serialized)

    def test_31_attempt_logs_preserve_treatment_and_prompt(self):
        params = self.params()
        self.runner(params).run()
        final = next(r for r in self.rows(params, "api_attempts.jsonl")
                     if r["stage"] == "final_decision")
        self.assertEqual(final["prompt_version"], "promise_switch_v1")
        self.assertIsNotNone(final["condition"])
        self.assertIn("messages", final["request"]["body"])

    def test_32_nebius_qwen_reasoning_envelope_is_withheld_not_repaired(self):
        clean, metadata = normalize_reasoning_content(
            '<think>hidden trace</think>\n{"message":"final answer"}',
            provider="nebius", model_id="Qwen/Qwen3-32B", reasoning_mode="on",
        )
        self.assertEqual(clean, '{"message":"final answer"}')
        self.assertTrue(metadata["reasoning_trace_withheld"])
        incomplete, incomplete_metadata = normalize_reasoning_content(
            "<think>truncated trace", provider="nebius",
            model_id="Qwen/Qwen3-32B", reasoning_mode="on",
        )
        self.assertEqual(incomplete, "[WITHHELD_INCOMPLETE_REASONING_TRACE]")
        self.assertTrue(incomplete_metadata["reasoning_trace_withheld"])
        self.assertTrue(incomplete_metadata["reasoning_trace_incomplete"])

    def test_32b_provider_reasoning_signal_is_recorded_without_hidden_text(self):
        hidden = "private hidden reasoning that must not be retained"
        with_signal = normalize_openai_response(
            raw=json.dumps({
                "id": "response-id",
                "choices": [{
                    "message": {
                        "content": '{"decision":"COOPERATE","justification":"brief"}',
                        "reasoning": hidden,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 12},
            }),
            http_status=200, headers={}, latency_seconds=0.1,
        )
        self.assertTrue(with_signal.error_metadata["reasoning_signal_present"])
        self.assertEqual(with_signal.error_metadata["reasoning_signal_fields"], ["reasoning"])
        self.assertNotIn(hidden, json.dumps(with_signal.to_dict()))
        without_signal = normalize_openai_response(
            raw=json.dumps({
                "choices": [{
                    "message": {
                        "content": '{"decision":"DEFECT","justification":"brief"}',
                        "reasoning_content": "  ",
                    },
                    "finish_reason": "stop",
                }],
            }),
            http_status=200, headers={}, latency_seconds=0.1,
        )
        self.assertFalse(without_signal.error_metadata["reasoning_signal_present"])
        self.assertEqual(without_signal.error_metadata["reasoning_signal_fields"], [])


class WriterLockTests(ExecutionTestCase):
    def test_33_first_writer_acquires_lock_with_safe_metadata(self):
        params = self.params(dry_run=True)
        with RunWriterLock(params.run_dir, run_id=params.run_id) as lock:
            metadata = lock.metadata()
            self.assertEqual(metadata["run_id"], params.run_id)
            self.assertEqual(metadata["pid"], os.getpid())
            self.assertNotIn("secret", json.dumps(metadata).lower())

    def test_34_second_new_writer_is_rejected_before_journal_creation(self):
        params = self.params(dry_run=True)
        with RunWriterLock(params.run_dir, run_id=params.run_id):
            with self.assertRaises(RunLockError):
                PromiseSwitchRunner(params).run()
            self.assertFalse(any((params.run_dir / name).exists() for name in FILES))

    def test_35_second_resume_writer_is_rejected_with_zero_mutations(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        before = self._journal_snapshot(params)
        with RunWriterLock(params.run_dir, run_id=params.run_id):
            with self.assertRaises(RunLockError):
                PromiseSwitchRunner(params).run(resume=True)
        self.assertEqual(before, self._journal_snapshot(params))

    def test_36_released_lock_allows_resume(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        lock = RunWriterLock(params.run_dir, run_id=params.run_id).acquire()
        lock.release()
        summary = PromiseSwitchRunner(params).run(resume=True)
        self.assertEqual(summary["run_status"], "COMPLETED")

    def test_37_process_crash_releases_os_lock(self):
        params = self.params(dry_run=True)
        ready = params.run_dir / "child-ready"
        code = (
            "import sys,time; from pathlib import Path; "
            "from promise_switch.execution.lock import RunWriterLock; "
            "d=Path(sys.argv[1]); lock=RunWriterLock(d,run_id='run').acquire(); "
            "(d/'child-ready').write_text('ready'); time.sleep(30)"
        )
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", code, str(params.run_dir)],
            cwd=Path(__file__).resolve().parents[1],
        )
        try:
            for _ in range(200):
                if ready.exists():
                    break
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            with self.assertRaises(RunLockError):
                RunWriterLock(params.run_dir, run_id=params.run_id).acquire()
            process.kill()
            process.wait(timeout=5)
            with RunWriterLock(params.run_dir, run_id=params.run_id):
                pass
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_38_different_run_ids_lock_independently(self):
        first = self.params(run_id="first", dry_run=True)
        second = self.params(run_id="second", dry_run=True)
        with RunWriterLock(first.run_dir, run_id=first.run_id):
            with RunWriterLock(second.run_dir, run_id=second.run_id):
                self.assertNotEqual(first.run_dir, second.run_dir)

    def test_39_inspector_reads_while_writer_lock_exists(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        root = Path(__file__).resolve().parents[1]
        with RunWriterLock(params.run_dir, run_id=params.run_id):
            result = subprocess.run(
                [sys.executable, "-B", "scripts/inspect_promise_switch_run.py",
                 str(params.run_dir)],
                cwd=root, check=False, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("[manifest.json]", result.stdout)

    def test_40_stage3b_concurrent_resume_process_rejected_without_duplicates(self):
        params = self.params(dry_run=True, n=2)
        PromiseSwitchRunner(params).run()
        before = self._journal_snapshot(params)
        code = (
            "import sys; from pathlib import Path; "
            "from promise_switch.execution.runner import parameters_from_manifest,PromiseSwitchRunner; "
            "p=parameters_from_manifest(Path(sys.argv[1])); "
            "PromiseSwitchRunner(p).run(resume=True)"
        )
        root = Path(__file__).resolve().parents[1]
        with RunWriterLock(params.run_dir, run_id=params.run_id):
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, str(params.run_dir)],
                cwd=root, check=False, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another Promise Switch writer", result.stderr)
            self.assertEqual(before, self._journal_snapshot(params))
        attempts = self.rows(params, "api_attempts.jsonl")
        trials = self.rows(params, "trials.jsonl")
        self.assertEqual(len({row["attempt_id"] for row in attempts}), len(attempts))
        self.assertEqual(len({row["trial_id"] for row in trials}), len(trials))

    def test_41_reasoning_token_budget_is_manifested_and_immutable_on_resume(self):
        reasoning_on = create_model_config(
            provider="nebius", model_alias="qwen3_32b", reasoning_mode="on",
        )
        params = self.params(
            dry_run=True, model_configs=(reasoning_on,),
            message_max_tokens=4096, decision_max_tokens=4096,
        )
        PromiseSwitchRunner(params).run()
        manifest = RunStorage(params.run_dir).read_manifest()
        self.assertEqual(manifest["max_output_tokens"], 4096)
        self.assertEqual(manifest["token_limits"]["message"], 4096)
        changed = ExecutionParameters(**{
            **params.__dict__, "message_max_tokens": 2048,
            "decision_max_tokens": 2048,
            "output_cap_override_authorized": True,
        })
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PromiseSwitchRunner(changed).run(resume=True)

    def test_42_invalid_marker_is_visible_to_inspector(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        (params.run_dir / "INVALID_CONCURRENT_EXECUTION.json").write_text(
            '{"artifact_status":"INVALID_CONCURRENT_EXECUTION"}\n', encoding="utf-8"
        )
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-B", "scripts/inspect_promise_switch_run.py",
             str(params.run_dir)],
            cwd=root, check=True, capture_output=True, text=True,
        )
        self.assertIn("[INVALID RUN MARKER]", result.stdout)
        self.assertIn("INVALID_CONCURRENT_EXECUTION", result.stdout)

    def test_43_duplicate_journal_rows_fail_integrity_before_resume(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        storage = RunStorage(params.run_dir)
        duplicate = storage.read_jsonl("assignments.jsonl")[0]
        storage.append("assignments.jsonl", duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_run_integrity(params.run_dir)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PromiseSwitchRunner(params).run(resume=True)

    def test_43b_corrupted_terminal_action_fails_integrity(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        trials_path = params.run_dir / "trials.jsonl"
        rows = RunStorage(params.run_dir).read_jsonl("trials.jsonl")
        rows[0]["final_action"] = "DEFECT"
        trials_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "mechanically parsed final response"):
            validate_run_integrity(params.run_dir)
        with self.assertRaisesRegex(ValueError, "mechanically parsed final response"):
            PromiseSwitchRunner(params).run(resume=True)

    def test_44_confirmatory_role_requires_approved_n_and_real_transport(self):
        with self.assertRaisesRegex(ValueError, "n=100"):
            self.params(dataset_role="confirmatory", n=99)
        with self.assertRaisesRegex(ValueError, "dry_run"):
            self.params(dataset_role="confirmatory", n=100, dry_run=True)
        params = self.params(dataset_role="confirmatory", n=100)
        self.assertEqual(params.dataset_role, "confirmatory")
        self.assertEqual(params.temperature, 1.0)

    def test_44b_temperature_defaults_to_one_and_override_is_explicitly_manifested(self):
        params = self.params(dry_run=True)
        PromiseSwitchRunner(params).run()
        manifest = RunStorage(params.run_dir).read_manifest()
        self.assertEqual(manifest["temperature"], 1.0)
        self.assertFalse(manifest["temperature_override_authorized"])
        request_params = self.params(run_id="temperature-requests")
        transport = FakeTransport()
        self.runner(request_params, transport).run()
        self.assertTrue(transport.calls)
        self.assertTrue(all(call.body["temperature"] == 1.0 for call in transport.calls))
        with self.assertRaisesRegex(ValueError, "temperature.*authorized override"):
            self.params(run_id="silent-zero", dry_run=True, temperature=0.0)
        historical_style = self.params(
            run_id="authorized-zero", dry_run=True, temperature=0.0,
            temperature_override_authorized=True,
        )
        PromiseSwitchRunner(historical_style).run()
        override_manifest = RunStorage(historical_style.run_dir).read_manifest()
        self.assertEqual(override_manifest["temperature"], 0.0)
        self.assertTrue(override_manifest["temperature_override_authorized"])

    def test_45_n100_plan_and_voluntary_allocation_are_offline_and_exact(self):
        first = plan_trials(
            run_id="offline-confirmatory-fixture", seed=20260811,
            n_per_condition=100, model_configs=(self.config,),
        )
        second = plan_trials(
            run_id="offline-confirmatory-fixture", seed=20260811,
            n_per_condition=100, model_configs=(self.config,),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 600)
        voluntary = [plan for plan in first if plan.planned_regime == "voluntary"]
        self.assertEqual(len(voluntary), 200)
        fixed_counts = Counter(
            plan.final_condition for plan in first if plan.final_condition is not None
        )
        self.assertEqual(set(fixed_counts.values()), {100})
        events = [MessageEvent(
            message_event_id=f"event:{plan.trial_id}", trial_id=plan.trial_id,
            timestamp="2026-08-11T00:00:00Z", raw_message=f"content-{index}",
            message_recorded=True,
        ) for index, plan in enumerate(voluntary)]
        assignments = allocate_voluntary_pool(seed=17, message_events=events)
        changed_content = [MessageEvent(
            message_event_id=event.message_event_id, trial_id=event.trial_id,
            timestamp=event.timestamp, raw_message="changed", message_recorded=True,
        ) for event in events]
        repeated = allocate_voluntary_pool(seed=17, message_events=changed_content)
        self.assertEqual(
            [(row.trial_id, row.condition) for row in assignments],
            [(row.trial_id, row.condition) for row in repeated],
        )
        voluntary_counts = Counter(row.condition for row in assignments)
        self.assertEqual(set(voluntary_counts.values()), {100})
        self.assertEqual(len({row.trial_id for row in assignments}), 200)
        states = {plan.trial_id: TrialState.PLANNED for plan in first}
        conditions = {plan.trial_id: plan.final_condition for plan in first}
        self.assertEqual(len(balanced_pending_order(first, states, conditions)), 600)
        self.assertEqual(len(json.loads(json.dumps({
            key: value.value for key, value in states.items()
        }))), 600)

    def test_45b_temporary_confirmatory_manifest_has_frozen_effective_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            params = ExecutionParameters(
                run_id="temporary-confirmatory-plan", output_dir=output,
                n_per_condition=100, seed=20260811,
                model_configs=(self.config,), dataset_role="confirmatory",
                rate_limit_policy=RateLimitPolicy(
                    requests_per_minute=None, tokens_per_minute=None,
                    max_concurrency=1, minimum_interval_seconds=0,
                ),
            )
            transport = FakeTransport()
            runner = PromiseSwitchRunner(params, transport=transport)
            runner._create_run()
            manifest = RunStorage(params.run_dir).read_manifest()
            plans = RunStorage(params.run_dir).read_jsonl("planned_trials.jsonl")
            estimate = runner.preflight_estimate()["model_configurations"][0]
            full_estimate = runner.preflight_estimate()
            self.assertEqual(manifest["dataset_role"], "confirmatory")
            self.assertEqual(
                manifest["requested_completed_trials_per_final_condition"], 100,
            )
            self.assertEqual(len(estimate["target_final_cells"]), 6)
            self.assertEqual(len(plans), 600)
            self.assertEqual(sum(row["planned_regime"] == "voluntary" for row in plans), 200)
            self.assertEqual(manifest["temperature"], 1.0)
            self.assertFalse(manifest["temperature_override_authorized"])
            self.assertEqual(manifest["max_output_tokens"], 4096)
            self.assertEqual(manifest["prompt_version"], "promise_switch_v1")
            self.assertEqual(manifest["prompt_hash"], prompt_templates_hash())
            self.assertEqual(full_estimate["dataset_role"], "confirmatory")
            self.assertEqual(full_estimate["temperature"], 1.0)
            self.assertEqual(full_estimate["token_limits"]["final_decision"], 4096)
            self.assertEqual(full_estimate["prompt_version"], "promise_switch_v1")
            self.assertEqual(full_estimate["prompt_hash"], prompt_templates_hash())
            self.assertEqual(transport.calls, [])
            temporary_run_dir = params.run_dir
        self.assertFalse(temporary_run_dir.exists())

    def test_46_condition_label_leakage_stops_before_transport(self):
        params = self.params()
        transport = FakeTransport()
        with patch(
            "promise_switch.execution.runner.voluntary_message_messages",
            return_value=[{"role": "user", "content": "Voluntary Message--Switch"}],
        ):
            with self.assertRaisesRegex(RuntimeError, "condition-label leakage"):
                self.runner(params, transport).run()
        self.assertEqual(transport.calls, [])

    def test_47_model_caps_resolve_by_configuration_and_low_cap_needs_authorization(self):
        expected = {
            ("nebius", "qwen3_32b", "off"): 4096,
            ("nebius", "qwen3_32b", "on"): 4096,
            ("nebius", "llama_3_3_70b", "off"): 256,
            ("nebius", "gpt_oss_120b_nebius", "on"): 2048,
            ("nebius", "deepseek_v4_pro_nebius_thinking_on", "on"): 2048,
            ("nebius", "deepseek_v4_pro_nebius_thinking_off", "off"): 2048,
            ("nebius", "deepseek_v4_flash_nebius_thinking_on", "on"): 2048,
            ("nebius", "deepseek_v4_flash_nebius_thinking_off", "off"): 2048,
            ("nebius", "kimi_k3_nebius", "unavailable"): 2048,
            ("nebius", "minimax_m3_nebius_thinking_on", "on"): 2048,
            ("nebius", "minimax_m3_nebius_thinking_off", "off"): 2048,
            ("nebius_uk", "glm_5_2_nebius_thinking_on", "on"): 2048,
            ("nebius_uk", "glm_5_2_nebius_thinking_off", "off"): 2048,
            ("groq", "gpt_oss_120b_groq", "on"): 2048,
        }
        for index, ((provider, alias, mode), cap) in enumerate(expected.items()):
            with self.subTest(provider=provider, alias=alias, mode=mode):
                config = create_model_config(
                    provider=provider, model_alias=alias, reasoning_mode=mode,
                )
                params = self.params(
                    run_id=f"cap-{index}", model_configs=(config,), dry_run=True,
                )
                self.assertEqual(params.message_max_tokens, cap)
                self.assertEqual(params.decision_max_tokens, cap)

        kimi = create_model_config(
            provider="nebius", model_alias="kimi_k3_nebius",
        )
        with self.assertRaisesRegex(ValueError, "authorized override"):
            self.params(
                run_id="unsafe-kimi", model_configs=(kimi,), dry_run=True,
                message_max_tokens=256, decision_max_tokens=256,
            )
        overridden = self.params(
            run_id="authorized-kimi", model_configs=(kimi,), dry_run=True,
            message_max_tokens=256, decision_max_tokens=256,
            output_cap_override_authorized=True,
        )
        PromiseSwitchRunner(overridden).run()
        manifest = RunStorage(overridden.run_dir).read_manifest()
        self.assertEqual(manifest["token_limits"], {
            "message": 256, "final_decision": 256,
        })
        self.assertTrue(manifest["output_cap_override_authorized"])

    def test_47b_new_matrix_arms_manifest_effective_identity_and_sampling(self):
        cases = (
            ("nebius", "minimax_m3_nebius_thinking_on", "on"),
            ("nebius", "minimax_m3_nebius_thinking_off", "off"),
            ("nebius_uk", "glm_5_2_nebius_thinking_on", "on"),
            ("nebius_uk", "glm_5_2_nebius_thinking_off", "off"),
            ("nebius", "deepseek_v4_pro_nebius_thinking_on", "on"),
            ("nebius", "deepseek_v4_pro_nebius_thinking_off", "off"),
            ("nebius", "deepseek_v4_flash_nebius_thinking_on", "on"),
            ("nebius", "deepseek_v4_flash_nebius_thinking_off", "off"),
            ("groq", "gpt_oss_120b_groq", "on"),
        )
        for index, (provider, alias, mode) in enumerate(cases):
            with self.subTest(provider=provider, alias=alias):
                config = create_model_config(
                    provider=provider, model_alias=alias, reasoning_mode=mode,
                )
                params = self.params(
                    run_id=f"manifest-new-config-{index}",
                    model_configs=(config,), dry_run=True,
                )
                PromiseSwitchRunner(params).run()
                manifest = RunStorage(params.run_dir).read_manifest()
                row = manifest["model_configurations"][0]
                self.assertEqual(row["provider"], provider)
                self.assertEqual(row["model_id"], config.model_id)
                self.assertEqual(row["reasoning_mode"], mode)
                self.assertEqual(
                    row["production_max_output_tokens"],
                    config.production_max_output_tokens,
                )
                self.assertEqual(manifest["temperature"], 1.0)
                self.assertEqual(
                    manifest["max_output_tokens"],
                    config.production_max_output_tokens,
                )
                self.assertEqual(manifest["prompt_version"], "promise_switch_v1")
                self.assertEqual(manifest["prompt_hash"], prompt_templates_hash())

    @staticmethod
    def _journal_snapshot(params):
        return {
            name: (params.run_dir / name).read_bytes()
            for name in FILES if (params.run_dir / name).exists()
        }


class CompletedCellTopUpTests(ExecutionTestCase):
    def top_up_params(self, run_id: str, *, n: int = 2) -> ExecutionParameters:
        return self.params(run_id=run_id, n=n, dry_run=True, top_up_enabled=True)

    def test_48_no_attrition_requires_no_top_up(self):
        params = self.top_up_params("no-attrition")
        summary = self.runner(params, FakeTransport()).run()
        self.assertEqual(summary["run_status"], "COMPLETED")
        self.assertTrue(summary["completion_target_met"])
        self.assertEqual(summary["top_up_trials_added"], 0)
        self.assertEqual(read_jsonl(params.run_dir / "top_up_events.jsonl"), [])

    def test_49_fixed_behavioral_attrition_adds_new_trial_and_preserves_failure(self):
        params = self.top_up_params("fixed-behavioral")
        transport = OneFailureTransport(Condition.NO_COMMUNICATION)
        summary = self.runner(params, transport).run()
        trials = read_jsonl(params.run_dir / "trials.jsonl")
        plans = read_jsonl(params.run_dir / "planned_trials.jsonl")
        failures = [row for row in trials if row.get("failure_category") == "behavioral"]
        top_ups = [row for row in plans if row["plan_phase"] == "top_up"]
        self.assertEqual(summary["run_status"], "COMPLETED")
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(top_ups), 1)
        self.assertNotEqual(failures[0]["trial_id"], top_ups[0]["trial_id"])
        self.assertEqual(len({row["trial_id"] for row in plans}), len(plans))
        counts = summary["counts_by_model_provider_reasoning_condition"]
        cell = next(value for key, value in counts.items() if key.endswith("|no_communication"))
        self.assertEqual(cell["completed_binary"], 2)
        self.assertEqual(cell["behavioral_missing_action"], 1)
        self.assertEqual(cell["top_up_trials_added"], 1)
        result = binary_contrast(trials, __import__(
            "promise_switch.analysis", fromlist=["PRIMARY_CONFIRMATORY_CONTRASTS"]
        ).PRIMARY_CONFIRMATORY_CONTRASTS[0])
        self.assertEqual(result["attrition_sensitivity"]["reference_missing_action_n"], 1)

    def test_50_infrastructure_attrition_pauses_then_tops_up(self):
        params = self.top_up_params("fixed-infrastructure")
        transport = OneFailureTransport(Condition.NO_COMMUNICATION, infrastructure=True)
        first = self.runner(params, transport).run()
        self.assertEqual(first["run_status"], "PAUSED_INFRASTRUCTURE_REVIEW")
        self.assertEqual(first["top_up_trials_added"], 0)
        final = self.runner(params, transport).run(resume=True)
        self.assertEqual(final["run_status"], "COMPLETED")
        self.assertEqual(final["top_up_trials_added"], 1)
        trials = read_jsonl(params.run_dir / "trials.jsonl")
        self.assertEqual(sum(row.get("failure_category") == "infrastructure" for row in trials), 1)
        self.assertEqual(len(read_jsonl(params.run_dir / "pause_events.jsonl")), 1)

    def test_51_voluntary_asymmetric_attrition_uses_balanced_block_and_overshoots(self):
        params = self.top_up_params("voluntary-overshoot")
        transport = OneFailureTransport(Condition.VOLUNTARY_PROMISE_SWITCH)
        summary = self.runner(params, transport).run()
        plans = read_jsonl(params.run_dir / "planned_trials.jsonl")
        events = read_jsonl(params.run_dir / "top_up_events.jsonl")
        top_up_voluntary = [row for row in plans
                            if row["plan_phase"] == "top_up" and row["planned_regime"] == "voluntary"]
        self.assertEqual(len(top_up_voluntary), 2)
        self.assertEqual(len({row["top_up_block_id"] for row in top_up_voluntary}), 1)
        self.assertEqual(events[0]["voluntary_balanced_blocks"], 1)
        cells = summary["counts_by_model_provider_reasoning_condition"]
        no_switch = next(v for k, v in cells.items() if k.endswith("|voluntary_promise_no_switch"))
        switch = next(v for k, v in cells.items() if k.endswith("|voluntary_promise_switch"))
        self.assertEqual(no_switch["completed_binary"], 3)
        self.assertEqual(switch["completed_binary"], 2)
        self.assertEqual(switch["behavioral_missing_action"], 1)
        assignments = read_jsonl(params.run_dir / "assignments.jsonl")
        block_ids = {row["trial_id"] for row in top_up_voluntary}
        block_conditions = [row["condition"] for row in assignments if row["trial_id"] in block_ids]
        self.assertCountEqual(block_conditions, [
            Condition.VOLUNTARY_PROMISE_NO_SWITCH.value,
            Condition.VOLUNTARY_PROMISE_SWITCH.value,
        ])
        allocation_seed = next(
            row["randomization_seed"] for row in assignments if row["trial_id"] in block_ids
        )
        changed_content = [MessageEvent(
            message_event_id=f"changed:{index}", trial_id=row["trial_id"], timestamp="t",
            raw_message="completely different" if index else "", message_recorded=True,
        ) for index, row in enumerate(top_up_voluntary)]
        changed_assignments = allocate_voluntary_pool(
            seed=allocation_seed, message_events=changed_content,
        )
        self.assertEqual(
            {row.trial_id: row.condition.value for row in changed_assignments},
            {row["trial_id"]: row["condition"] for row in assignments
             if row["trial_id"] in block_ids},
        )
        result = binary_contrast(
            read_jsonl(params.run_dir / "trials.jsonl"), VOLUNTARY_ITT_CONTRAST,
        )
        self.assertEqual(result["treatment_n"], 3)
        self.assertEqual(result["reference_n"], 2)
        self.assertEqual(result["attrition_sensitivity"]["reference_missing_action_n"], 1)
        audit = audit_run(params.run_dir)
        self.assertTrue(audit["minimum_completed_target_met"])
        self.assertEqual(audit["rows"]["top_up_plans"], 2)
        self.assertEqual(audit["final_actions"]["None"], 1)

    def test_52_top_up_planning_is_blind_to_cooperate_or_defect(self):
        snapshots = []
        for index, action in enumerate(("COOPERATE", "DEFECT")):
            with tempfile.TemporaryDirectory() as directory:
                params = ExecutionParameters(**{
                    **self.top_up_params("outcome-blind").__dict__,
                    "output_dir": Path(directory),
                })
                self.runner(
                    params, OneFailureTransport(Condition.NO_COMMUNICATION, action=action),
                ).run()
                events = read_jsonl(params.run_dir / "top_up_events.jsonl")
                snapshots.append([{key: value for key, value in event.items() if key != "timestamp"}
                                  for event in events])
        self.assertEqual(snapshots[0], snapshots[1])

    def test_53_crash_resume_during_top_up_is_idempotent(self):
        for index, crash_event in enumerate((
            "before_top_up_creation", "after_top_up_event", "after_top_up_plan_persistence",
            "after_message_event", "after_assignment", "after_api_attempt_final",
        )):
            with self.subTest(event=crash_event):
                params = self.top_up_params(f"topup-crash-{index}")
                voluntary = crash_event in {"after_message_event", "after_assignment"}
                transport = OneFailureTransport(
                    Condition.VOLUNTARY_PROMISE_SWITCH if voluntary
                    else Condition.NO_COMMUNICATION,
                )

                class TopUpCrash:
                    def __init__(self):
                        self.ids: set[str] = set()
                        self.fired = False

                    def __call__(self, event, payload):
                        if event == "after_top_up_event":
                            self.ids.update(row["trial_id"] for row in payload["planned_trials"])
                        eligible = event == crash_event and (
                            crash_event in {"before_top_up_creation", "after_top_up_event",
                                            "after_top_up_plan_persistence"}
                            or payload.get("trial_id") in self.ids
                        )
                        if eligible and not self.fired:
                            self.fired = True
                            raise RuntimeError("simulated top-up crash")

                crash = TopUpCrash()
                with self.assertRaisesRegex(RuntimeError, "simulated top-up crash"):
                    self.runner(params, transport, hook=crash).run()
                final = self.runner(params, transport).run(resume=True)
                self.assertEqual(final["run_status"], "COMPLETED")
                plans = read_jsonl(params.run_dir / "planned_trials.jsonl")
                assignments = read_jsonl(params.run_dir / "assignments.jsonl")
                trials = read_jsonl(params.run_dir / "trials.jsonl")
                self.assertEqual(len({row["trial_id"] for row in plans}), len(plans))
                self.assertEqual(len({row["trial_id"] for row in assignments}), len(assignments))
                self.assertEqual(len({row["trial_id"] for row in trials}), len(trials))

    def test_54_quota_during_top_up_pauses_without_new_expansion(self):
        params = self.top_up_params("topup-quota")
        base = OneFailureTransport(Condition.NO_COMMUNICATION)
        top_up_ids: set[str] = set()
        quota_sent = False

        class QuotaTransport:
            calls = base.calls

            def complete(self, request):
                nonlocal quota_sent
                if request.metadata["trial_id"] in top_up_ids and not quota_sent:
                    quota_sent = True
                    return NormalizedResponse(
                        status="error", raw_response='{"error":"quota exhausted"}', http_status=429,
                    )
                return base.complete(request)

        def hook(event, payload):
            if event == "after_top_up_event":
                top_up_ids.update(row["trial_id"] for row in payload["planned_trials"])

        first = self.runner(params, QuotaTransport(), hook=hook).run()
        self.assertEqual(first["run_status"], "PAUSED_PROVIDER_QUOTA")
        self.assertEqual(first["top_up_events"], 1)
        final = self.runner(params, QuotaTransport(), hook=hook).run(resume=True)
        self.assertEqual(final["run_status"], "COMPLETED")
        self.assertEqual(final["top_up_events"], 1)

    def test_55_confirmatory_analysis_gate_requires_minimum_100(self):
        params = self.params(
            run_id="confirmatory-gate", n=100, dataset_role="confirmatory",
        )
        self.runner(params, FakeTransport()).run()
        validate_run_integrity(
            params.run_dir, required_dataset_role="confirmatory", require_completed=True,
        )
        path = params.run_dir / "trials.jsonl"
        rows = read_jsonl(path)
        rows[0].update({
            "completion_status": "failed", "final_action": None,
            "failure_category": "behavioral", "failure_reason": "invalid_output_format",
        })
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "minimum completed binary target"):
            validate_run_integrity(
                params.run_dir, required_dataset_role="confirmatory", require_completed=True,
            )

    def test_56_preassignment_infrastructure_failure_gets_message_recovery(self):
        params = self.top_up_params("message-infrastructure")
        transport = OneMessageInfrastructureFailureTransport()
        first = self.runner(params, transport).run()
        self.assertEqual(first["run_status"], "PAUSED_INFRASTRUCTURE_REVIEW")
        final = self.runner(params, transport).run(resume=True)
        self.assertEqual(final["run_status"], "COMPLETED")
        events = read_jsonl(params.run_dir / "top_up_events.jsonl")
        recovery = [row for row in events if row["event_type"] == "voluntary_message_recovery"]
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0]["message_recovery_count"], 1)
        plans = read_jsonl(params.run_dir / "planned_trials.jsonl")
        recovery_plan = next(row for row in plans
                             if row["creation_reason"] == "voluntary_message_stage_infrastructure_recovery")
        failed = next(row for row in read_jsonl(params.run_dir / "trials.jsonl")
                      if row.get("failure_category") == "infrastructure")
        self.assertNotEqual(recovery_plan["trial_id"], failed["trial_id"])
        self.assertEqual(len(read_jsonl(params.run_dir / "assignments.jsonl")), 12)
        validate_run_integrity(params.run_dir, require_completed=True)

    def test_57_corrupted_top_up_decision_is_rejected(self):
        params = self.top_up_params("corrupt-topup")
        self.runner(params, OneFailureTransport(Condition.NO_COMMUNICATION)).run()
        path = params.run_dir / "top_up_events.jsonl"
        events = read_jsonl(path)
        events[0]["completed_binary_counts_before"][Condition.NO_COMMUNICATION.value] = 2
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completion-only stopping rule"):
            validate_run_integrity(params.run_dir)

    def test_58_reviewed_behavioral_class_does_not_repeat_pause_or_change_top_up(self):
        params = self.top_up_params("several-behavioral", n=4)
        transport = SeveralBehavioralFailuresTransport(Condition.NO_COMMUNICATION, 3)
        summary = self.runner(params, transport).run()
        self.assertEqual(summary["run_status"], "PAUSED_BEHAVIORAL_PARSER_REVIEW")
        failure_class = summary["pause_detail"]["parser_failure_class"]
        summary = self.runner(params, transport).run(
            resume=True, approve_parser_review_class=failure_class,
        )
        self.assertEqual(summary["run_status"], "COMPLETED")
        trials = read_jsonl(params.run_dir / "trials.jsonl")
        plans = read_jsonl(params.run_dir / "planned_trials.jsonl")
        failures = [row for row in trials if row.get("failure_category") == "behavioral"]
        fixed_topups = [row for row in plans if row["plan_phase"] == "top_up"
                        and row["final_condition"] == Condition.NO_COMMUNICATION.value]
        self.assertEqual(len(failures), 3)
        self.assertEqual(len(fixed_topups), 3)
        self.assertEqual(len(read_jsonl(params.run_dir / "pause_events.jsonl")), 1)
        self.assertEqual(len(read_jsonl(params.run_dir / "parser_review_events.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
