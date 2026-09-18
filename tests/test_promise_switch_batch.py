from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from promise_switch.design import Condition
from promise_switch.client import ProviderAgnosticClient, RequestSpec
from promise_switch.artifacts import validate_run_integrity
from promise_switch.coding.semantic_v2 import (
    judge_family_allowed, model_family, verify_frozen_codebook,
)
from promise_switch.execution.anthropic_batch import (
    AnthropicBatchProvider, _strip_hidden_reasoning,
)
from promise_switch.execution.attrition import (
    classify_batch_decision_response, classify_decision_output, classify_transport,
)
from promise_switch.execution.batch_transport import (
    BatchCoordinator, BatchLifecycleState, BatchSubmissionError,
    ProviderBatchStatus, deterministic_custom_id, make_mapping,
)
from promise_switch.execution.batch_identity import verify_production_batch_attempt
from promise_switch.execution.batch_runner import PromiseSwitchBatchRunner
from promise_switch.execution.batch_workflow import (
    allocate_recorded_voluntary_groups, build_decision_batch_requests,
    build_message_batch_requests,
)
from promise_switch.execution.openai_batch import OpenAIBatchProvider
from promise_switch.execution.openai_batch import _strip_openai_hidden_reasoning
from promise_switch.execution.planner import plan_trials
from promise_switch.future_runs import FUTURE_CONFIGURATIONS, audit_future_run
from promise_switch.execution.retry import RetryPolicy
from promise_switch.execution.runner import ExecutionParameters, PromiseSwitchRunner
from promise_switch.execution.lock import RunLockError, RunWriterLock
from promise_switch.providers import create_model_config
from promise_switch.prompts import final_decision_messages, voluntary_message_messages
from promise_switch.parser import parse_decision
from promise_switch.design import treatment_metadata
from promise_switch.randomization import assign_fixed_trial, plan_fixed_trial
from promise_switch.schemas import AssignmentRecord, MessageEvent, NormalizedResponse
from promise_switch.version import RESPONSE_QC_VERSION, prompt_templates_hash
from scripts.prepare_promise_switch_semantic_v2 import main as prepare_semantic_v2
from scripts.preflight_promise_switch_batch import (
    build_report as build_preflight_report, main as preflight_main,
    model_identity_verification, parser as preflight_parser, reasoning_verification,
)
from scripts.run_promise_switch_batch import main as batch_cli_main
from scripts.finalize_promise_switch_semantic_v2 import main as finalize_semantic_v2
from scripts.prepare_promise_switch_validation_supplement import main as prepare_validation
from scripts.analyze_promise_switch_final import (
    BASE_RUN_SPECS, RUN_SPECS, activated_specs,
)
from scripts.run_promise_switch_message_judge import main as semantic_judge_main


def configured(provider: str, alias: str, mode: str, cap: int = 128):
    return replace(
        create_model_config(provider=provider, model_alias=alias, reasoning_mode=mode),
        production_max_output_tokens=cap,
    )


class FakeProvider:
    name = "fake"

    def __init__(self, results=None, ambiguous=False, status=None):
        self.results = results or []
        self.ambiguous = ambiguous
        self.provider_status = status or ProviderBatchStatus("completed", True, True, {})

    def serialize(self, mapping):
        return {"custom_id": mapping.custom_id, "body": mapping.request.body}

    def submit(self, input_path, *, metadata):
        if self.ambiguous:
            raise BatchSubmissionError("unknown", known_not_submitted=False)
        return "batch_123"

    def poll(self, provider_job_id):
        return self.provider_status

    def retrieve(self, provider_job_id, status):
        return self.results

    def normalize(self, result):
        if "response" in result:
            return NormalizedResponse(**result["response"])
        return NormalizedResponse(status="success", raw_response=result.get("text", ""))


class FakeAnthropicProvider(FakeProvider):
    name = "anthropic"

    def serialize(self, mapping):
        return AnthropicBatchProvider().serialize(mapping)


class RegistryTests(unittest.TestCase):
    def test_four_exact_rows(self):
        expected = {
            ("openai", "gpt_5_6_luna_openai", "off"): ("gpt-5.6-luna", {"reasoning_effort": "none"}, "max_completion_tokens", "openai_batch_chat_completions"),
            ("openai", "gpt_5_6_luna_openai", "on"): ("gpt-5.6-luna", {"reasoning_effort": "medium"}, "max_completion_tokens", "openai_batch_chat_completions"),
            ("anthropic", "claude_sonnet_5_anthropic", "off"): ("claude-sonnet-5", {"thinking": {"type": "disabled"}}, "max_tokens", "anthropic_message_batches"),
            ("anthropic", "claude_sonnet_5_anthropic", "on"): ("claude-sonnet-5", {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}}, "max_tokens", "anthropic_message_batches"),
        }
        for key, values in expected.items():
            row = create_model_config(provider=key[0], model_alias=key[1], reasoning_mode=key[2])
            self.assertEqual((row.model_id, row.provider_request_kwargs, row.output_token_field, row.batch_protocol), values)
            self.assertIsNone(row.production_max_output_tokens)
            self.assertTrue(row.batch_only_confirmatory)

    def test_unresolved_cap_and_synchronous_confirmatory_are_rejected(self):
        row = create_model_config(provider="openai", model_alias="gpt_5_6_luna_openai", reasoning_mode="off")
        with self.assertRaisesRegex(ValueError, "Batch-only"):
            ExecutionParameters("r", Path("x"), 100, 20260808, (replace(row, production_max_output_tokens=1),), "confirmatory")
        with self.assertRaisesRegex(ValueError, "unresolved"):
            ExecutionParameters("r", Path("x"), 100, 20260808, (row,), "confirmatory", execution_mode="provider_batch")


class SerializationTests(unittest.TestCase):
    def _requests(self, config):
        plans = plan_trials(run_id="r", seed=20260808, n_per_condition=1, model_configs=(config,))
        voluntary = next(plan for plan in plans if plan.planned_regime == "voluntary")
        return build_message_batch_requests([voluntary], config=config, max_tokens=128)[0]

    def test_openai_exact_envelope_and_prompt_order(self):
        for mode, effort in (("off", "none"), ("on", "medium")):
            request = self._requests(configured("openai", "gpt_5_6_luna_openai", mode))
            envelope = OpenAIBatchProvider().serialize(make_mapping(request, 1))
            self.assertEqual(envelope["url"], "/v1/chat/completions")
            self.assertEqual([x["role"] for x in envelope["body"]["messages"]], ["system", "user"])
            self.assertEqual(envelope["body"]["reasoning_effort"], effort)
            self.assertEqual(envelope["body"]["temperature"], 1.0)
            self.assertIn("max_completion_tokens", envelope["body"])
            self.assertNotIn("max_tokens", envelope["body"])
            self.assertFalse({"tools", "response_format"} & envelope["body"].keys())
        self.assertEqual(prompt_templates_hash(), "17b5f31eaac9d9f2742d74a01fae17bc4a93a847d33f49d1016392035a8f175c")

    def test_anthropic_mapping_and_reasoning(self):
        for mode in ("off", "on"):
            request = self._requests(configured("anthropic", "claude_sonnet_5_anthropic", mode))
            envelope = AnthropicBatchProvider().serialize(make_mapping(request, 1))
            params = envelope["params"]
            self.assertEqual(params["system"], request.body["messages"][0]["content"])
            self.assertEqual(params["messages"], [request.body["messages"][1]])
            self.assertEqual(params["temperature"], 1.0)
            self.assertIn("max_tokens", params)
            if mode == "off":
                self.assertEqual(params["thinking"], {"type": "disabled"})
                self.assertNotIn("output_config", params)
            else:
                self.assertEqual(params["thinking"], {"type": "adaptive", "display": "omitted"})
                self.assertEqual(params["output_config"], {"effort": "xhigh"})

    def test_normalizers_ignore_order_and_strip_hidden_reasoning(self):
        row = {"custom_id": "x", "result": {"type": "succeeded", "message": {
            "id": "msg", "content": [
                {"type": "thinking", "thinking": "secret", "signature": "sig"},
                {"type": "text", "text": '{"decision":"COOPERATE","justification":"x"}'},
            ], "stop_reason": "end_turn", "usage": {"output_tokens_details": {"thinking_tokens": 7}},
        }}}
        cleaned = _strip_hidden_reasoning(row)
        serialized = json.dumps(cleaned)
        self.assertNotIn("secret", serialized)
        self.assertNotIn("signature", serialized)
        normalized = AnthropicBatchProvider().normalize(cleaned)
        self.assertEqual(normalized.finish_reason, "stop")
        self.assertEqual(normalized.error_metadata["thinking_token_count"], 7)

        openai = {"custom_id": "y", "response": {"status_code": 200, "body": {
            "id": "resp", "choices": [{"message": {
                "content": "visible", "reasoning_content": "hidden"
            }, "finish_reason": "stop"}],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 5}},
        }}}
        openai_cleaned = _strip_openai_hidden_reasoning(openai)
        self.assertNotIn("hidden", json.dumps(openai_cleaned))
        openai_response = OpenAIBatchProvider().normalize(openai_cleaned)
        self.assertEqual(openai_response.raw_response, "visible")
        self.assertEqual(openai_response.error_metadata["reasoning_token_count"], 5)


class ClaudeProductionIdentityTests(unittest.TestCase):
    @staticmethod
    def _attempt(mode: str = "off", cap: int = 256):
        config = configured("anthropic", "claude_sonnet_5_anthropic", mode, cap)
        request = ProviderAgnosticClient(config).build_request(
            messages=[{"role": "system", "content": "system"},
                      {"role": "user", "content": "user"}],
            temperature=1.0, max_tokens=cap,
        ).to_log_dict()
        return config, request

    def test_returned_model_must_be_present_and_exact(self):
        config, request = self._attempt()
        accepted = verify_production_batch_attempt(
            config=config, request=request,
            response=NormalizedResponse(
                "success", "{}", returned_model_id="claude-sonnet-5",
                error_metadata={"thinking_present": False, "thinking_token_count": 0},
            ),
        )
        self.assertTrue(accepted.verified)
        for returned_model in (None, "claude-sonnet-5-latest"):
            rejected = verify_production_batch_attempt(
                config=config, request=request,
                response=NormalizedResponse(
                    "success", "{}", returned_model_id=returned_model,
                    error_metadata={"thinking_present": False, "thinking_token_count": 0},
                ),
            )
            self.assertFalse(rejected.verified)
            self.assertEqual(rejected.failure_cause, "model_identity_failure")
        # The configured/requested model cannot substitute for absent provider evidence.
        self.assertEqual(request["model_id"], config.model_id)

    def test_off_configuration_and_telemetry_are_fail_closed(self):
        config, request = self._attempt("off", 256)
        response = NormalizedResponse(
            "success", "{}", returned_model_id="claude-sonnet-5",
            error_metadata={"thinking_present": False, "thinking_token_count": 0},
        )
        mutations = (
            ("thinking", {"type": "adaptive", "display": "omitted"}),
            ("max_tokens", 255),
            ("temperature", 0.0),
        )
        for field, value in mutations:
            altered = json.loads(json.dumps(request))
            altered["body"][field] = value
            check = verify_production_batch_attempt(
                config=config, request=altered, response=response,
            )
            self.assertFalse(check.verified, field)
            self.assertEqual(check.failure_cause, "request_configuration_identity_failure")
        telemetry_failure = verify_production_batch_attempt(
            config=config, request=request,
            response=replace(response, error_metadata={
                "thinking_present": True, "thinking_token_count": 1,
            }),
        )
        self.assertFalse(telemetry_failure.verified)

    def test_adaptive_configuration_is_verified_independently_of_token_count(self):
        config, request = self._attempt("on", 4096)
        for count in (0, 37):
            check = verify_production_batch_attempt(
                config=config, request=request,
                response=NormalizedResponse(
                    "success", "{}", returned_model_id="claude-sonnet-5",
                    error_metadata={
                        "thinking_present": bool(count), "thinking_token_count": count,
                    },
                ),
            )
            self.assertTrue(check.verified)
        for field, value in (
            ("thinking", {"type": "disabled"}),
            ("output_config", {"effort": "high"}),
        ):
            altered = json.loads(json.dumps(request))
            altered["body"][field] = value
            self.assertFalse(verify_production_batch_attempt(
                config=config, request=altered,
                response=NormalizedResponse(
                    "success", "{}", returned_model_id="claude-sonnet-5",
                ),
            ).verified)


class WorkflowTests(unittest.TestCase):
    def test_assignment_boundary_and_exact_message_in_fresh_context(self):
        config = configured("openai", "gpt_5_6_luna_openai", "off")
        plans = plan_trials(run_id="r", seed=20260808, n_per_condition=1, model_configs=(config,))
        voluntary = [plan for plan in plans if plan.planned_regime == "voluntary"]
        self.assertEqual(allocate_recorded_voluntary_groups(plans, seed=20260808, message_events={}), ())
        with self.assertRaises(ValueError):
            build_message_batch_requests(voluntary, config=config, max_tokens=128, assignments={voluntary[0].trial_id: {}})
        events = {plan.trial_id: MessageEvent("event:" + plan.trial_id, plan.trial_id, "now", 'verbatim "message"', True) for plan in voluntary}
        assignments = allocate_recorded_voluntary_groups(plans, seed=20260808, message_events=events)
        self.assertEqual(len(assignments), 2)
        requests = build_decision_batch_requests(
            voluntary, config=config, max_tokens=128,
            assignments={row.trial_id: row for row in assignments}, message_events=events,
        )
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(len(request.body["messages"]) == 2 for request in requests))
        self.assertTrue(all('verbatim "message"' in request.body["messages"][1]["content"] for request in requests))

    def test_topup_group_repeats_same_boundary(self):
        config = configured("openai", "gpt_5_6_luna_openai", "off")
        from promise_switch.execution.planner import plan_top_up_expansion
        plans = plan_top_up_expansion(run_id="r", seed=1, config=config, sequence=1, fixed_deficits={}, voluntary_blocks=1, starting_ordinal=0)
        self.assertEqual(len(build_message_batch_requests(plans, config=config, max_tokens=128)), 2)
        self.assertEqual(allocate_recorded_voluntary_groups(plans, seed=1, message_events={}), ())

    def test_batch_prompt_bytes_equal_frozen_builder_for_every_treatment(self):
        config = configured("openai", "gpt_5_6_luna_openai", "off")
        plans = plan_trials(run_id="r", seed=20260808, n_per_condition=1, model_configs=(config,))
        voluntary_plan = next(plan for plan in plans if plan.planned_regime == "voluntary")
        message_request = build_message_batch_requests(
            [voluntary_plan], config=config, max_tokens=128,
        )[0]
        original = treatment_metadata(
            Condition.VOLUNTARY_PROMISE_NO_SWITCH, voluntary_plan.trial_id,
        ).original_partner_id
        self.assertEqual(
            json.dumps(message_request.body["messages"], ensure_ascii=False).encode(),
            json.dumps(voluntary_message_messages(original), ensure_ascii=False).encode(),
        )
        for condition in Condition:
            plan = next(row for row in plans if (
                row.final_condition == condition.value
                or condition in {Condition.VOLUNTARY_PROMISE_NO_SWITCH, Condition.VOLUNTARY_PROMISE_SWITCH}
                and row.planned_regime == "voluntary"
            ))
            metadata = treatment_metadata(condition, plan.trial_id)
            event = MessageEvent("event", plan.trial_id, "now", "exact raw message", True)
            assignment = AssignmentRecord(
                "assignment", plan.trial_id, plan.randomization_seed, condition,
                event.message_event_id if plan.planned_regime == "voluntary" else None,
                metadata.original_partner_id, metadata.final_partner_id, metadata.switched,
            )
            request = build_decision_batch_requests(
                [plan], config=config, max_tokens=128,
                assignments={plan.trial_id: assignment},
                message_events=({plan.trial_id: event} if plan.planned_regime == "voluntary" else {}),
            )[0]
            expected = final_decision_messages(
                condition=condition,
                original_partner_id=metadata.original_partner_id,
                final_partner_id=metadata.final_partner_id,
                raw_message="exact raw message" if plan.planned_regime == "voluntary" else "",
            )
            self.assertEqual(
                json.dumps(request.body["messages"], ensure_ascii=False).encode(),
                json.dumps(expected, ensure_ascii=False).encode(),
            )


class CoordinatorTests(unittest.TestCase):
    def _request(self):
        config = configured("openai", "gpt_5_6_luna_openai", "off")
        plan = next(plan for plan in plan_trials(run_id="r", seed=1, n_per_condition=1, model_configs=(config,)) if plan.planned_regime == "voluntary")
        return build_message_batch_requests([plan], config=config, max_tokens=128)[0]

    def _requests(self):
        config = configured("openai", "gpt_5_6_luna_openai", "off")
        plans = [plan for plan in plan_trials(
            run_id="r", seed=1, n_per_condition=1, model_configs=(config,)
        ) if plan.planned_regime == "voluntary"]
        return build_message_batch_requests(plans, config=config, max_tokens=128)

    def test_hashes_job_id_and_unordered_complete_results(self):
        request = self._request()
        mapping = make_mapping(request, 1)
        provider = FakeProvider([{"custom_id": mapping.custom_id, "text": "ok"}])
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(Path(directory), provider, timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            self.assertEqual(coordinator.submit(key), "batch_123")
            normalized = coordinator.poll_and_retrieve(key)
            self.assertEqual(normalized[mapping.custom_id].raw_response, "ok")
            coordinator.mark_ingested(key)
            jobs = coordinator.storage.read_jsonl("batch_jobs.jsonl")
            self.assertEqual(jobs[-1]["status"], BatchLifecycleState.INGESTED.value)
            self.assertTrue(jobs[-2]["result_artifact_hash"])
            self.assertEqual(jobs[2]["provider_job_id"], "batch_123")
            with self.assertRaises(ValueError):
                coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])

    def test_ambiguous_submit_crash_and_explicit_reconciliation(self):
        request = self._request()
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(Path(directory), FakeProvider(ambiguous=True), timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            with self.assertRaises(BatchSubmissionError):
                coordinator.submit(key)
            self.assertEqual(coordinator._latest(key)["status"], "RECONCILIATION_REQUIRED")
            with self.assertRaises(ValueError):
                coordinator.submit(key)
            coordinator.reconcile_submission(key, "recovered_batch")
            self.assertEqual(coordinator._latest(key)["provider_job_id"], "recovered_batch")

    def test_crash_marker_requires_explicit_reconciliation(self):
        request = self._request()
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(Path(directory), FakeProvider(), timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            current = coordinator._latest(key)
            coordinator._event(key, BatchLifecycleState.SUBMITTING, current)
            with self.assertRaises(ValueError):
                coordinator.submit(key)
            coordinator.mark_interrupted_submission_for_reconciliation(key)
            self.assertEqual(coordinator._latest(key)["status"], "RECONCILIATION_REQUIRED")

    def test_missing_results_are_preserved_for_partial_ingestion(self):
        request = self._request()
        mapping = make_mapping(request, 1)
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(Path(directory), FakeProvider([]), timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            coordinator.submit(key)
            self.assertEqual(coordinator.poll_and_retrieve(key), {})
            self.assertEqual(coordinator.latest(key)["missing_custom_ids"], [mapping.custom_id])
            coordinator.mark_ingested(key)
            self.assertEqual(coordinator.latest(key)["status"], "RECONCILIATION_REQUIRED")

    def test_duplicate_results_fail_closed(self):
        request = self._request()
        mapping = make_mapping(request, 1)
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(
                Path(directory), FakeProvider([
                    {"custom_id": mapping.custom_id}, {"custom_id": mapping.custom_id},
                ]), timestamp=lambda: "now",
            )
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            coordinator.submit(key)
            with self.assertRaises(ValueError):
                coordinator.poll_and_retrieve(key)
            self.assertEqual(coordinator.latest(key)["status"], "RECONCILIATION_REQUIRED")

    def test_same_attempt_cannot_cross_batch_sequences(self):
        request = self._request()
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(Path(directory), FakeProvider(), timestamp=lambda: "now")
            coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            with self.assertRaisesRegex(ValueError, "already exists"):
                coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=2, batch_phase="message", requests=[(request, 1)])

    def test_concurrent_submit_is_blocked_before_second_provider_call(self):
        request = self._request()
        entered = threading.Event()
        release = threading.Event()

        class BlockingProvider(FakeProvider):
            calls = 0

            def submit(self, input_path, *, metadata):
                self.calls += 1
                entered.set()
                release.wait(2)
                return "batch_once"

        with tempfile.TemporaryDirectory() as directory:
            provider = BlockingProvider()
            coordinator = BatchCoordinator(Path(directory), provider, timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            errors = []
            first = threading.Thread(target=lambda: coordinator.submit(key))
            first.start()
            self.assertTrue(entered.wait(1))
            try:
                coordinator.submit(key)
            except Exception as exc:
                errors.append(exc)
            release.set()
            first.join()
            self.assertEqual(provider.calls, 1)
            self.assertTrue(any(isinstance(exc, RunLockError) for exc in errors))

    def test_process_death_after_acceptance_never_auto_resubmits(self):
        request = self._request()

        class AcceptedThenDied(FakeProvider):
            calls = 0

            def submit(self, input_path, *, metadata):
                self.calls += 1
                raise SystemExit("process died after provider accepted")

        with tempfile.TemporaryDirectory() as directory:
            provider = AcceptedThenDied()
            coordinator = BatchCoordinator(Path(directory), provider, timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(request, 1)])
            with self.assertRaises(SystemExit):
                coordinator.submit(key)
            self.assertEqual(coordinator.latest(key)["status"], "SUBMITTING")
            with self.assertRaises(ValueError):
                coordinator.submit(key)
            self.assertEqual(provider.calls, 1)
            coordinator.mark_interrupted_submission_for_reconciliation(key)
            self.assertEqual(coordinator.latest(key)["status"], "RECONCILIATION_REQUIRED")

    def test_crash_before_submission_recovers_orphaned_prepare_operationally(self):
        request = self._request()
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(Path(directory), FakeProvider(), timestamp=lambda: "now")
            kwargs = dict(
                run_id="r", model="m", reasoning_mode="off", batch_sequence=1,
                batch_phase="message", requests=[(request, 1)],
            )
            with patch.object(coordinator, "_event", side_effect=SystemExit("crash")):
                with self.assertRaises(SystemExit):
                    coordinator.prepare(**kwargs)
            key = coordinator.prepare(**kwargs)
            self.assertTrue(coordinator.latest(key)["recovered_orphaned_prepare"])
            self.assertEqual(coordinator.submit(key), "batch_123")

    def test_truly_unordered_multi_request_results_and_partial_coverage(self):
        requests = self._requests()
        mappings = [make_mapping(request, 1) for request in requests]
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider([
                {"custom_id": mappings[1].custom_id, "text": "second"},
                {"custom_id": mappings[0].custom_id, "text": "first"},
            ])
            coordinator = BatchCoordinator(Path(directory), provider, timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(row, 1) for row in requests])
            coordinator.submit(key)
            results = coordinator.poll_and_retrieve(key)
            self.assertEqual(results[mappings[0].custom_id].raw_response, "first")
            self.assertEqual(results[mappings[1].custom_id].raw_response, "second")

        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider([{"custom_id": mappings[1].custom_id, "text": "known"}])
            coordinator = BatchCoordinator(Path(directory), provider, timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(row, 1) for row in requests])
            coordinator.submit(key)
            results = coordinator.poll_and_retrieve(key)
            self.assertEqual(set(results), {mappings[1].custom_id})
            self.assertEqual(coordinator.latest(key)["missing_custom_ids"], [mappings[0].custom_id])

    def test_whole_batch_expiry_yields_one_retryable_result_per_request(self):
        requests = self._requests()
        with tempfile.TemporaryDirectory() as directory:
            provider = FakeProvider(status=ProviderBatchStatus("expired", True, False, {}))
            coordinator = BatchCoordinator(Path(directory), provider, timestamp=lambda: "now")
            key = coordinator.prepare(run_id="r", model="m", reasoning_mode="off", batch_sequence=1, batch_phase="message", requests=[(row, 1) for row in requests])
            coordinator.submit(key)
            results = coordinator.poll_and_retrieve(key)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(classify_transport(row).retryable for row in results.values()))
            self.assertTrue(coordinator.latest(key)["whole_batch_unexecuted"])

    def test_results_retrieved_replays_from_hash_verified_disk_without_polling(self):
        requests = self._requests()
        mappings = [make_mapping(request, 1) for request in requests]
        provider = FakeProvider([
            {"custom_id": row.custom_id, "text": f"result-{index}"}
            for index, row in enumerate(mappings)
        ])
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            coordinator = BatchCoordinator(run_dir, provider, timestamp=lambda: "now")
            key = coordinator.prepare(
                run_id="r", model="m", reasoning_mode="off", batch_sequence=1,
                batch_phase="message", requests=[(row, 1) for row in requests],
            )
            coordinator.submit(key)
            first = coordinator.poll_and_retrieve(key)
            self.assertEqual(len(first or {}), 2)
            provider.poll = lambda *_: self.fail("replay contacted provider poll")
            provider.retrieve = lambda *_: self.fail("replay contacted provider retrieve")
            restarted = BatchCoordinator(run_dir, provider, timestamp=lambda: "later")
            replayed = restarted.poll_and_retrieve(key)
            self.assertEqual(
                {name: row.raw_response for name, row in (replayed or {}).items()},
                {name: row.raw_response for name, row in (first or {}).items()},
            )

    def test_replay_rejects_tampered_result_artifact(self):
        request = self._request()
        mapping = make_mapping(request, 1)
        with tempfile.TemporaryDirectory() as directory:
            coordinator = BatchCoordinator(
                Path(directory), FakeProvider([{"custom_id": mapping.custom_id, "text": "ok"}]),
                timestamp=lambda: "now",
            )
            key = coordinator.prepare(
                run_id="r", model="m", reasoning_mode="off", batch_sequence=1,
                batch_phase="message", requests=[(request, 1)],
            )
            coordinator.submit(key)
            coordinator.poll_and_retrieve(key)
            Path(coordinator.latest(key)["result_path"]).write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                coordinator.poll_and_retrieve(key)


class FailureAndAttemptsTests(unittest.TestCase):
    def test_attempt_record_persists_adaptive_config_and_observation_separately(self):
        config = configured("anthropic", "claude_sonnet_5_anthropic", "on", 4096)
        plan = plan_trials(
            run_id="r", seed=1, n_per_condition=1, model_configs=(config,),
        )[0]
        request = RequestSpec(
            base_url=config.base_url, model_id=config.model_id,
            api_key_env=config.api_key_env,
            body={
                "model": config.model_id,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "xhigh"},
            },
        )
        response = NormalizedResponse(
            "success", '{"decision":"COOPERATE","justification":"ok"}',
            returned_model_id=config.model_id,
            error_metadata={"thinking_present": False, "thinking_token_count": 0},
        )
        runner = object.__new__(PromiseSwitchRunner)
        runner.assignments = {}
        runner.timestamp = lambda: "now"
        outcome = classify_transport(response)
        row = runner._attempt_record(plan, "final_decision", 1, request, response, outcome)
        self.assertEqual(row["configured_reasoning_mode"], "adaptive/xhigh")
        self.assertFalse(row["thinking_observed"])
        self.assertEqual(row["thinking_tokens"], 0)
        self.assertEqual(row["transport_category"], "success")

    def test_transport_failure_policy_and_attempt_limit(self):
        self.assertEqual(RetryPolicy().max_attempts, 3)
        cases = [
            (NormalizedResponse("error", "", http_status=429), True, False),
            (NormalizedResponse("error", "quota billing", http_status=429), False, True),
            (NormalizedResponse("error", "", http_status=500), True, False),
            (NormalizedResponse("error", "", http_status=401), False, False),
            (NormalizedResponse("error", "", http_status=403), False, False),
            (NormalizedResponse("error", "", http_status=404), False, False),
            (NormalizedResponse("timeout", ""), True, False),
            (NormalizedResponse("malformed_api_response", ""), True, False),
            (NormalizedResponse("success", "", finish_reason="max_tokens"), True, False),
        ]
        for response, retryable, quota in cases:
            outcome = classify_transport(response)
            self.assertEqual((outcome.retryable, outcome.quota_pause), (retryable, quota))

    def test_qc_version_not_duplicated_in_adapter(self):
        self.assertEqual(RESPONSE_QC_VERSION, "promise_switch_response_qc_v3")
        self.assertFalse(hasattr(OpenAIBatchProvider(), "parse_decision"))
        self.assertFalse(hasattr(AnthropicBatchProvider(), "parse_decision"))

    def test_frozen_legacy_qc_ignores_provider_refusal_telemetry(self):
        parsed = parse_decision('{"decision":"COOPERATE","justification":"ok"}')
        legacy = classify_decision_output(parsed)
        batch = classify_batch_decision_response(
            NormalizedResponse("success", parsed.raw_response, finish_reason="refusal"),
            parsed,
        )
        self.assertEqual((legacy.category, legacy.cause), ("success", None))
        self.assertEqual((batch.category, batch.cause), ("behavioral", "refusal_to_decide"))

    def test_custom_id_is_deterministic_safe_and_attempt_bounded(self):
        first = deterministic_custom_id("ps:trial/raw text", "message", 1)
        self.assertEqual(first, deterministic_custom_id("ps:trial/raw text", "message", 1))
        self.assertRegex(first, r"^ps_[0-9a-f]{24}_m_a1$")
        self.assertNotIn("raw", first)
        with self.assertRaises(ValueError):
            deterministic_custom_id("x", "message", 4)

    def test_semantic_v2_family_policy_is_explicit(self):
        self.assertEqual(model_family("gpt-5.6-luna"), "gpt_5_6_luna")
        self.assertEqual(model_family("claude-sonnet-5"), "claude")
        self.assertEqual(model_family("openai", "some-other-model"), "openai")
        with self.assertRaises(ValueError):
            judge_family_allowed(
                source_family="gpt_5_6_luna", judge_family="gpt_oss",
                gpt_luna_policy=None,
            )
        self.assertTrue(judge_family_allowed(
            source_family="gpt_5_6_luna", judge_family="gpt_oss",
            gpt_luna_policy="narrow",
        ))
        self.assertFalse(judge_family_allowed(
            source_family="gpt_5_6_luna", judge_family="gpt_oss",
            gpt_luna_policy="conservative",
        ))


class BatchOrchestrationTests(unittest.TestCase):
    def _runner(
        self, directory: str, *, top_up: bool = False, provider_name: str = "openai",
        reasoning_mode: str = "off", cap: int = 128,
    ):
        alias = (
            "claude_sonnet_5_anthropic" if provider_name == "anthropic"
            else "gpt_5_6_luna_openai"
        )
        config = configured(provider_name, alias, reasoning_mode, cap)
        parameters = ExecutionParameters(
            "batch_run", Path(directory), 1, 20260808, (config,), "synthetic",
            execution_mode="provider_batch", dry_run=True, top_up_enabled=top_up,
        )
        provider = FakeAnthropicProvider() if provider_name == "anthropic" else FakeProvider()
        provider.name = provider_name
        runner = PromiseSwitchBatchRunner(parameters, provider, timestamp=lambda: "now")
        runner.initialize()
        return runner, provider, parameters

    @staticmethod
    def _results(runner, key, text):
        return [{"custom_id": row["custom_id"], "text": text} for row in runner.coordinator.mappings(key)]

    def _complete_claude_run(self, directory: str):
        runner, provider, parameters = self._runner(
            directory, provider_name="anthropic", cap=256,
        )
        for sequence, text in (
            (1, '{"message":"ok"}'),
            (2, '{"decision":"COOPERATE","justification":"ok"}'),
        ):
            key = runner.prepare_next_batch(batch_sequence=sequence)
            provider.results = [{
                "custom_id": row["custom_id"],
                "response": NormalizedResponse(
                    "success", text, returned_model_id="claude-sonnet-5",
                    finish_reason="stop", usage={"output_tokens": 12},
                    error_metadata={"thinking_present": False, "thinking_token_count": 0},
                ).to_dict(),
            } for row in runner.coordinator.mappings(key)]
            runner.coordinator.submit(key)
            runner.ingest_retrieved(key, runner.poll_and_retrieve(key) or {})
        self.assertIsNone(runner.prepare_next_batch(batch_sequence=3))
        return parameters.run_dir

    def test_complete_voluntary_durability_assignment_and_fresh_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, parameters = self._runner(directory)
            message_key = runner.prepare_next_batch(batch_sequence=1)
            self.assertIsNotNone(message_key)
            self.assertEqual(runner.coordinator.latest(message_key)["batch_phase"], "message")
            provider.results = self._results(runner, message_key, '{"message":"verbatim promise"}')
            runner.coordinator.submit(message_key)
            responses = runner.poll_and_retrieve(message_key)
            runner.ingest_retrieved(message_key, responses or {})
            events = runner.coordinator.storage.read_jsonl("message_events.jsonl")
            assignments = runner.coordinator.storage.read_jsonl("assignments.jsonl")
            event_ids = {row["message_event_id"] for row in events}
            voluntary = [row for row in assignments if row.get("message_event_id")]
            self.assertEqual(len(voluntary), 2)
            self.assertTrue(all(row["message_event_id"] in event_ids for row in voluntary))

            decision_key = runner.prepare_next_batch(batch_sequence=2)
            decision_mappings = runner.coordinator.mappings(decision_key)
            voluntary_ids = {row["trial_id"] for row in voluntary}
            voluntary_requests = [row for row in decision_mappings if row["trial_id"] in voluntary_ids]
            self.assertTrue(all(
                "verbatim promise" in row["request"]["body"]["messages"][1]["content"]
                for row in voluntary_requests
            ))
            provider.results = self._results(
                runner, decision_key, '{"decision":"COOPERATE","justification":"ok"}',
            )
            provider.results[0] = {
                "custom_id": decision_mappings[0]["custom_id"],
                "response": NormalizedResponse(
                    "success", '{"decision":"COOPERATE","justification":"ok"}',
                    provider_request_id="req_batch_persisted",
                ).to_dict(),
            }
            runner.coordinator.submit(decision_key)
            responses = runner.poll_and_retrieve(decision_key)
            runner.ingest_retrieved(decision_key, responses or {})
            self.assertIsNone(runner.prepare_next_batch(batch_sequence=3))
            self.assertEqual(len(runner.coordinator.storage.read_jsonl("trials.jsonl")), 6)
            self.assertIn(
                "req_batch_persisted",
                [row["response"].get("provider_request_id") for row in
                 runner.coordinator.storage.read_jsonl("api_attempts.jsonl")],
            )

    def test_retry_batches_increment_only_failed_attempt_and_stop_at_three(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, _ = self._runner(directory)
            key = runner.prepare_next_batch(batch_sequence=1)
            mappings = runner.coordinator.mappings(key)
            failed_id = mappings[0]["custom_id"]
            provider.results = [
                {"custom_id": row["custom_id"], "response": (
                    NormalizedResponse("error", "", http_status=500).to_dict()
                    if row["custom_id"] == failed_id
                    else NormalizedResponse("success", '{"message":"ok"}').to_dict()
                )} for row in mappings
            ]
            runner.coordinator.submit(key)
            runner.ingest_retrieved(key, runner.poll_and_retrieve(key) or {})
            for sequence, expected_attempt in ((2, 2), (3, 3)):
                retry_key = runner.prepare_next_batch(batch_sequence=sequence)
                retry_mappings = runner.coordinator.mappings(retry_key)
                self.assertEqual(len(retry_mappings), 1)
                self.assertEqual(retry_mappings[0]["attempt_number"], expected_attempt)
                self.assertEqual(
                    retry_mappings[0]["request_body_hash"], mappings[0]["request_body_hash"],
                )
                self.assertEqual(retry_mappings[0]["reasoning_mode"], mappings[0]["reasoning_mode"])
                provider.results = [{
                    "custom_id": retry_mappings[0]["custom_id"],
                    "response": NormalizedResponse("error", "", http_status=500).to_dict(),
                }]
                runner.coordinator.submit(retry_key)
                runner.ingest_retrieved(retry_key, runner.poll_and_retrieve(retry_key) or {})
            with self.assertRaisesRegex(RuntimeError, "paused"):
                runner.prepare_next_batch(batch_sequence=4)
            attempts = [row for row in runner.coordinator.storage.read_jsonl("api_attempts.jsonl")
                        if row["trial_id"] == mappings[0]["trial_id"]]
            self.assertEqual([row["attempt_number"] for row in attempts], [1, 2, 3])

    def test_partial_ingestion_retries_known_5xx_but_not_missing_request(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, _ = self._runner(directory)
            key = runner.prepare_next_batch(batch_sequence=1)
            mappings = runner.coordinator.mappings(key)
            known = mappings[0]
            missing = mappings[1]
            provider.results = [{
                "custom_id": known["custom_id"],
                "response": NormalizedResponse("error", "", http_status=500).to_dict(),
            }]
            runner.coordinator.submit(key)
            responses = runner.poll_and_retrieve(key) or {}
            self.assertEqual(set(responses), {known["custom_id"]})
            runner.ingest_retrieved(key, responses)
            self.assertEqual(runner.coordinator.latest(key)["status"], "RECONCILIATION_REQUIRED")
            retry_key = runner.prepare_next_batch(batch_sequence=2)
            retry = runner.coordinator.mappings(retry_key)
            self.assertEqual([(row["trial_id"], row["attempt_number"]) for row in retry], [
                (known["trial_id"], 2),
            ])
            self.assertNotEqual(retry[0]["trial_id"], missing["trial_id"])

    def test_voluntary_top_up_starts_with_message_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, _ = self._runner(directory, top_up=True)
            key = runner.prepare_next_batch(batch_sequence=1)
            provider.results = self._results(runner, key, '{"message":"ok"}')
            runner.coordinator.submit(key)
            runner.ingest_retrieved(key, runner.poll_and_retrieve(key) or {})
            key = runner.prepare_next_batch(batch_sequence=2)
            mappings = runner.coordinator.mappings(key)
            assignment_by_trial = {
                row["trial_id"]: row for row in
                runner.coordinator.storage.read_jsonl("assignments.jsonl")
            }
            voluntary_id = next(row["trial_id"] for row in mappings
                                if assignment_by_trial[row["trial_id"]].get("message_event_id"))
            provider.results = [{"custom_id": row["custom_id"], "text": (
                "not json" if row["trial_id"] == voluntary_id
                else '{"decision":"COOPERATE","justification":"ok"}'
            )} for row in mappings]
            runner.coordinator.submit(key)
            runner.ingest_retrieved(key, runner.poll_and_retrieve(key) or {})
            topup_key = runner.prepare_next_batch(batch_sequence=3)
            self.assertEqual(runner.coordinator.latest(topup_key)["batch_phase"], "message")
            topup_trials = {
                row["trial_id"] for row in
                runner.coordinator.storage.read_jsonl("planned_trials.jsonl")
                if row["plan_phase"] == "top_up"
            }
            self.assertTrue(all(row["trial_id"] in topup_trials
                                for row in runner.coordinator.mappings(topup_key)))

    def test_mixed_success_parser_5xx_429_refusal_and_expiry_are_individual(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, _ = self._runner(directory)
            message_key = runner.prepare_next_batch(batch_sequence=1)
            provider.results = self._results(runner, message_key, '{"message":"ok"}')
            runner.coordinator.submit(message_key)
            runner.ingest_retrieved(message_key, runner.poll_and_retrieve(message_key) or {})
            key = runner.prepare_next_batch(batch_sequence=2)
            mappings = runner.coordinator.mappings(key)
            responses = [
                NormalizedResponse("success", '{"decision":"COOPERATE","justification":"ok"}'),
                NormalizedResponse("success", "not json"),
                NormalizedResponse("error", "", http_status=500),
                NormalizedResponse("error", "quota billing exhausted", http_status=429),
                NormalizedResponse("success", "", finish_reason="refusal"),
                NormalizedResponse("expired", "", error_metadata={"unexecuted": True}),
            ]
            provider.results = [
                {"custom_id": row["custom_id"], "response": response.to_dict()}
                for row, response in zip(mappings, responses, strict=True)
            ]
            runner.coordinator.submit(key)
            outcomes = runner.ingest_retrieved(key, runner.poll_and_retrieve(key) or {})
            self.assertEqual(set(outcomes.values()), {
                "completed", "terminal_behavioral", "retry_pending", "paused_quota",
            })
            attempts = runner.coordinator.storage.read_jsonl("api_attempts.jsonl")
            self.assertEqual(len([row for row in attempts if row["stage"] == "final_decision"]), 6)
            with self.assertRaises((RuntimeError, ValueError)):
                runner.prepare_next_batch(batch_sequence=3)

    def test_mid_ingestion_crash_replays_remaining_results_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, parameters = self._runner(directory)
            key = runner.prepare_next_batch(batch_sequence=1)
            provider.results = self._results(runner, key, '{"message":"durable"}')
            runner.coordinator.submit(key)
            responses = runner.poll_and_retrieve(key) or {}
            original = PromiseSwitchRunner.ingest_batch_response
            calls = {"count": 0}

            def crash_on_second(instance, *args, **kwargs):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise SystemExit("simulated crash")
                return original(instance, *args, **kwargs)

            with patch.object(PromiseSwitchRunner, "ingest_batch_response", new=crash_on_second):
                with self.assertRaises(SystemExit):
                    runner.ingest_retrieved(key, responses)
            self.assertEqual(runner.coordinator.latest(key)["status"], "RESULTS_RETRIEVED")
            provider.poll = lambda *_: self.fail("restart contacted provider")
            provider.retrieve = lambda *_: self.fail("restart contacted provider")
            restarted = PromiseSwitchBatchRunner(parameters, provider, timestamp=lambda: "later")
            replayed = restarted.poll_and_retrieve(key) or {}
            restarted.ingest_retrieved(key, replayed)
            storage = restarted.coordinator.storage
            counts = tuple(len(storage.read_jsonl(name)) for name in (
                "api_attempts.jsonl", "message_events.jsonl", "assignments.jsonl",
            ))
            self.assertEqual(counts, (2, 2, 6))
            restarted.ingest_retrieved(key, restarted.poll_and_retrieve(key) or {})
            self.assertEqual(tuple(len(storage.read_jsonl(name)) for name in (
                "api_attempts.jsonl", "message_events.jsonl", "assignments.jsonl",
            )), counts)

    def test_claude_model_mismatch_is_recorded_and_pauses_without_behavioral_use(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, provider, _ = self._runner(directory, provider_name="anthropic", cap=256)
            key = runner.prepare_next_batch(batch_sequence=1)
            mapping = runner.coordinator.mappings(key)[0]
            provider.results = [{
                "custom_id": mapping["custom_id"],
                "response": NormalizedResponse(
                    "success", '{"message":"must not be allocated"}',
                    returned_model_id="wrong-model",
                    error_metadata={"thinking_present": False, "thinking_token_count": 0},
                ).to_dict(),
            }]
            runner.coordinator.submit(key)
            outcomes = runner.ingest_retrieved(key, runner.poll_and_retrieve(key) or {})
            self.assertEqual(outcomes[mapping["custom_id"]], "terminal_infrastructure")
            attempts = runner.coordinator.storage.read_jsonl("api_attempts.jsonl")
            self.assertEqual(attempts[0]["identity_verification"], "FAILED")
            self.assertTrue(any("returned_model_id mismatch" in item
                                for item in attempts[0]["identity_failures"]))
            self.assertEqual(
                runner.coordinator.storage.read_jsonl("message_events.jsonl"), [],
            )
            self.assertEqual(
                runner.coordinator.storage.read_state()["run_status"],
                "PAUSED_INFRASTRUCTURE_REVIEW",
            )

    def test_artifact_audit_accepts_clean_claude_identity_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._complete_claude_run(directory)
            self.assertEqual(
                validate_run_integrity(run_dir, require_completed=True)["run_id"],
                "batch_run",
            )

        for mutation in ("wrong_model", "missing_model", "wrong_config"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                run_dir = self._complete_claude_run(directory)
                attempts_path = run_dir / "api_attempts.jsonl"
                attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
                if mutation == "wrong_model":
                    attempts[0]["response"]["returned_model_id"] = "wrong-model"
                elif mutation == "missing_model":
                    attempts[0]["response"]["returned_model_id"] = None
                else:
                    attempts[0]["request"]["body"]["thinking"] = {
                        "type": "adaptive", "display": "omitted",
                    }
                attempts_path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempts),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    validate_run_integrity(run_dir, require_completed=True)


class ProviderLifecycleTests(unittest.TestCase):
    class Response:
        def __init__(self, value):
            self.value = value
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def read(self):
            return self.value

    def test_realistic_openai_envelope_request_id_and_mocked_lifecycle(self):
        result_row = {"custom_id": "x", "response": {"request_id": "req_batch_42", "status_code": 200, "body": {
            "id": "chatcmpl_1", "model": "gpt-5.6-luna", "choices": [{"message": {"content": "visible"}, "finish_reason": "stop"}],
        }}}
        calls = []
        def fake_urlopen(request, **kwargs):
            calls.append((request.method, request.full_url))
            if request.full_url.endswith("/files") and request.method == "POST":
                return self.Response(b'{"id":"file_1"}')
            if request.full_url.endswith("/batches") and request.method == "POST":
                return self.Response(b'{"id":"batch_1"}')
            if request.full_url.endswith("/batches/batch_1"):
                return self.Response(b'{"status":"completed","output_file_id":"out_1"}')
            return self.Response((json.dumps(result_row) + "\n").encode())
        provider = OpenAIBatchProvider(urlopen=fake_urlopen)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            path = Path(directory) / "in.jsonl"
            path.write_text('{}\n')
            job = provider.submit(path, metadata={})
            status = provider.poll(job)
            rows = provider.retrieve(job, status)
        normalized = provider.normalize(rows[0])
        self.assertEqual(normalized.provider_request_id, "req_batch_42")
        self.assertEqual(normalized.returned_model_id, "gpt-5.6-luna")
        self.assertEqual(len(calls), 4)

    def test_mocked_anthropic_lifecycle(self):
        calls = []
        row = {"custom_id": "x", "result": {"type": "succeeded", "message": {
            "id": "msg_1", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "visible"}],
            "stop_reason": "end_turn", "usage": {},
        }}}
        def fake_urlopen(request, **kwargs):
            calls.append((request.method, request.full_url))
            if request.full_url.endswith("/messages/batches"):
                return self.Response(b'{"id":"batch_a"}')
            if request.full_url.endswith("/messages/batches/batch_a"):
                return self.Response(b'{"processing_status":"ended"}')
            return self.Response((json.dumps(row) + "\n").encode())
        provider = AnthropicBatchProvider(urlopen=fake_urlopen)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            path = Path(directory) / "in.jsonl"
            path.write_text('{"custom_id":"x","params":{}}\n')
            job = provider.submit(path, metadata={})
            status = provider.poll(job)
            result = provider.normalize(provider.retrieve(job, status)[0])
        self.assertEqual(result.provider_response_id, "msg_1")
        self.assertEqual(result.returned_model_id, "claude-sonnet-5")
        self.assertEqual(len(calls), 3)

    def test_returned_model_identity_match_mismatch_and_missing_for_both_providers(self):
        openai_body = lambda model: {"custom_id": "x", "response": {
            "status_code": 200, "body": {
                "id": "r", "model": model,
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            },
        }}
        anthropic_body = lambda model: {"custom_id": "x", "result": {
            "type": "succeeded", "message": {
                "id": "m", "model": model, "content": [{"type": "text", "text": "{}"}],
                "stop_reason": "end_turn", "usage": {},
            },
        }}
        for provider, builder, expected in (
            (OpenAIBatchProvider(), openai_body, "gpt-5.6-luna"),
            (AnthropicBatchProvider(), anthropic_body, "claude-sonnet-5"),
        ):
            matching = provider.normalize(builder(expected)).returned_model_id
            mismatch = provider.normalize(builder("wrong-model")).returned_model_id
            missing_row = builder(None)
            missing_row["response"]["body"].pop("model", None) if "response" in missing_row else missing_row["result"]["message"].pop("model", None)
            missing = provider.normalize(missing_row).returned_model_id
            self.assertEqual(model_identity_verification(expected, [matching]), "VERIFIED_MATCH")
            self.assertEqual(model_identity_verification(expected, [mismatch]), "MISMATCH")
            self.assertEqual(model_identity_verification(expected, [missing]), "UNVERIFIABLE")


class PreflightAndSemanticTests(unittest.TestCase):
    @staticmethod
    def _future_manifest(label="GPT-5.6 Luna — no-think"):
        expected = next(row for row in FUTURE_CONFIGURATIONS if row["label"] == label)
        return {
            "run_id": "future_run", "prompt_version": "promise_switch_v1",
            "prompt_hash": prompt_templates_hash(),
            "response_qc_version": "promise_switch_response_qc_v3",
            "requested_completed_trials_per_final_condition": 100,
            "model_configurations": [{
                name: expected[name]
                for name in ("model_id", "model_alias", "provider", "reasoning_mode")
            }],
        }

    def test_future_run_gate_rejects_incomplete_wrong_prompt_qc_and_identity(self):
        manifest = self._future_manifest()
        with patch("promise_switch.future_runs.validate_run_integrity", side_effect=ValueError("not completed")):
            with self.assertRaisesRegex(ValueError, "not completed"):
                audit_future_run("GPT-5.6 Luna — no-think", Path("future_run"))
        for field, value, message in (
            ("prompt_hash", "wrong", "prompt hash"),
            ("response_qc_version", "wrong", "QC version"),
        ):
            bad = {**manifest, field: value}
            with patch("promise_switch.future_runs.validate_run_integrity", return_value=bad):
                with self.assertRaisesRegex(ValueError, message):
                    audit_future_run("GPT-5.6 Luna — no-think", Path("future_run"))
        bad = json.loads(json.dumps(manifest))
        bad["model_configurations"][0]["model_id"] = "wrong"
        with patch("promise_switch.future_runs.validate_run_integrity", return_value=bad):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                audit_future_run("GPT-5.6 Luna — no-think", Path("future_run"))

    def test_minimal_preflight_contains_exactly_one_fixed_decision_request(self):
        args = preflight_parser().parse_args([
            "--provider", "openai", "--model", "gpt_5_6_luna_openai",
            "--reasoning-mode", "off", "--candidate-cap", "128",
        ])
        report, _, requests = build_preflight_report(args)
        self.assertEqual(report["submitted_request_count"], 1)
        self.assertEqual(report["request_count"], 1)
        self.assertEqual(len(report["serialized_requests"]), 1)
        self.assertEqual(len({report["serialized_requests"][0]["custom_id"]}), 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].metadata["stage"], "final_decision")
        self.assertNotEqual(requests[0].metadata["stage"], "message")

    def test_analysis_defaults_to_13_and_activation_requires_a_completed_audited_run(self):
        self.assertEqual(len(BASE_RUN_SPECS), 13)
        self.assertEqual(RUN_SPECS, BASE_RUN_SPECS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activate.json"
            run_path = Path(directory) / "future_run"
            path.write_text(json.dumps({"runs": [{
                "label": "GPT-5.6 Luna — no-think", "run_path": str(run_path),
            }]}))
            manifest = {
                "prompt_hash": prompt_templates_hash(),
                "response_qc_version": "promise_switch_response_qc_v3",
                "requested_completed_trials_per_final_condition": 100,
                "model_configurations": [{
                    "model_id": "gpt-5.6-luna", "reasoning_mode": "off",
                    "provider": "openai",
                }],
            }
            with patch(
                "scripts.analyze_promise_switch_final.validate_run_integrity",
                return_value=manifest,
            ) as audit:
                activated = activated_specs(path)
            self.assertEqual(activated[0][0], str(run_path.resolve()))
            audit.assert_called_once_with(
                run_path.resolve(), required_dataset_role="confirmatory",
                require_completed=True,
            )

    def test_anthropic_reasoning_verification_distinguishes_config_from_telemetry(self):
        def check(*, signal=False, count=0, transport="success", qc="success"):
            return [{
                "reasoning_signal_present": signal,
                "reasoning_token_count": count,
                "transport_status": transport,
                "qc_category": qc,
            }]

        def verify(configuration, checks, *, mode="on", identity="VERIFIED_MATCH"):
            return reasoning_verification(
                provider="anthropic", mode=mode,
                request_configuration=configuration, checks=checks,
                model_identity_status=identity,
            )

        disabled = verify({"thinking": {"type": "disabled"}}, check(), mode="off")
        self.assertEqual(disabled["reasoning_verification"], "VERIFIED_OFF")
        self.assertFalse(disabled["thinking_observed"])
        self.assertEqual(disabled["thinking_tokens"], 0)

        adaptive = {
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "xhigh"},
        }
        positive = verify(adaptive, check(signal=True, count=37))
        self.assertEqual(positive["reasoning_verification"], "CONFIG_VERIFIED_ADAPTIVE")
        self.assertTrue(positive["thinking_observed"])
        self.assertEqual(positive["thinking_tokens"], 37)

        zero = verify(adaptive, check())
        self.assertEqual(zero["reasoning_verification"], "CONFIG_VERIFIED_ADAPTIVE")
        self.assertFalse(zero["thinking_observed"])
        self.assertEqual(zero["thinking_tokens"], 0)

        missing = verify({"output_config": {"effort": "xhigh"}}, check())
        wrong_effort = verify({
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "high"},
        }, check())
        wrong_model = verify(adaptive, check(), identity="MISMATCH")
        rejected = verify(adaptive, check(transport="error", qc="infrastructure"))
        for result in (missing, wrong_effort, wrong_model, rejected):
            self.assertEqual(result["reasoning_verification"], "CONFIG_VERIFICATION_FAILED")

    def test_openai_reasoning_on_still_requires_positive_telemetry(self):
        base = {
            "transport_status": "success", "qc_category": "success",
            "reasoning_signal_present": False, "reasoning_token_count": 0,
        }
        absent = reasoning_verification(
            provider="openai", mode="on",
            request_configuration={"reasoning_effort": "medium"},
            checks=[base], model_identity_status="VERIFIED_MATCH",
        )
        present = reasoning_verification(
            provider="openai", mode="on",
            request_configuration={"reasoning_effort": "medium"},
            checks=[{**base, "reasoning_signal_present": True, "reasoning_token_count": 1}],
            model_identity_status="VERIFIED_MATCH",
        )
        self.assertEqual(absent["reasoning_verification"], "UNVERIFIABLE")
        self.assertEqual(present["reasoning_verification"], "VERIFIED_ON")

    def test_anthropic_adaptive_zero_telemetry_preflight_passes_offline(self):
        class OfflineCoordinator:
            def __init__(self, *args, **kwargs):
                pass
            def poll_and_retrieve(self, key):
                return {"d": NormalizedResponse(
                    "success", '{"decision":"COOPERATE","justification":"ok"}',
                    returned_model_id="claude-sonnet-5",
                    error_metadata={"thinking_present": False, "thinking_token_count": 0},
                )}
            def mappings(self, key):
                return ({"custom_id": "d", "stage": "final_decision"},)
            def latest(self, key):
                return {"status": "RESULTS_RETRIEVED", "missing_custom_ids": []}
            def mark_ingested(self, key):
                pass

        args = [
            "--provider", "anthropic", "--model", "claude_sonnet_5_anthropic",
            "--reasoning-mode", "on", "--candidate-cap", "4096",
            "--run-id", "p", "--poll-existing", "key",
            "--acknowledge-paid-provider-call", "I AUTHORIZE ONE PAID BATCH PREFLIGHT",
        ]
        with patch(
            "scripts.preflight_promise_switch_batch.BatchCoordinator", OfflineCoordinator,
        ), patch("builtins.print") as printer:
            self.assertEqual(preflight_main(args), 0)
        report = json.loads(printer.call_args.args[0])
        self.assertEqual(report["reasoning_verification"], "CONFIG_VERIFIED_ADAPTIVE")
        self.assertFalse(report["thinking_observed"])
        self.assertEqual(report["thinking_tokens"], 0)

    def test_preflight_cli_pass_and_failed_reasoning_exit_status(self):
        class OfflineCoordinator:
            def __init__(self, *args, **kwargs):
                self.marked = False
            def poll_and_retrieve(self, key):
                return {
                    "d": NormalizedResponse(
                        "success", '{"decision":"COOPERATE","justification":"ok"}',
                        returned_model_id="gpt-5.6-luna",
                    ),
                }
            def mappings(self, key):
                return ({"custom_id": "d", "stage": "final_decision"},)
            def _mappings(self, key):
                return list(self.mappings(key))
            def mark_ingested(self, key):
                self.marked = True
            def latest(self, key):
                return {"status": "RESULTS_RETRIEVED"}
        base = [
            "--provider", "openai", "--model", "gpt_5_6_luna_openai",
            "--candidate-cap", "128", "--run-id", "preflight", "--poll-existing", "key",
            "--acknowledge-paid-provider-call", "I AUTHORIZE ONE PAID BATCH PREFLIGHT",
        ]
        with patch("scripts.preflight_promise_switch_batch.BatchCoordinator", OfflineCoordinator), patch("builtins.print"):
            self.assertEqual(preflight_main([*base, "--reasoning-mode", "off"]), 0)
            self.assertNotEqual(preflight_main([*base, "--reasoning-mode", "on"]), 0)

    def test_paid_call_acknowledgement_is_mandatory(self):
        with self.assertRaisesRegex(SystemExit, "acknowledge-paid-provider-call"):
            preflight_main([
                "--provider", "openai", "--model", "gpt_5_6_luna_openai",
                "--reasoning-mode", "off", "--candidate-cap", "128",
                "--run-id", "preflight", "--live-submit",
            ])

    def test_preflight_model_mismatch_and_missing_identity_fail_closed(self):
        class Coordinator:
            returned_model = "wrong"
            def __init__(self, *args, **kwargs):
                pass
            def poll_and_retrieve(self, key):
                return {"d": NormalizedResponse(
                    "success", '{"decision":"COOPERATE","justification":"ok"}',
                    returned_model_id=self.returned_model,
                )}
            def mappings(self, key):
                return ({"custom_id": "d", "stage": "final_decision"},)
            def latest(self, key):
                return {"status": "RESULTS_RETRIEVED", "missing_custom_ids": []}
            def mark_ingested(self, key):
                self.marked = True
        args = [
            "--provider", "openai", "--model", "gpt_5_6_luna_openai",
            "--reasoning-mode", "off", "--candidate-cap", "128",
            "--run-id", "p", "--poll-existing", "key",
            "--acknowledge-paid-provider-call", "I AUTHORIZE ONE PAID BATCH PREFLIGHT",
        ]
        with patch("scripts.preflight_promise_switch_batch.BatchCoordinator", Coordinator), patch("builtins.print"):
            self.assertNotEqual(preflight_main(args), 0)
            Coordinator.returned_model = None
            self.assertNotEqual(preflight_main(args), 0)

    def test_one_request_preflight_valid_on_passes_with_reasoning_and_model(self):
        class Coordinator:
            submits = 0
            def __init__(self, *args, **kwargs):
                pass
            def poll_and_retrieve(self, key):
                return {"d": NormalizedResponse(
                    "success", '{"decision":"COOPERATE","justification":"ok"}',
                    returned_model_id="gpt-5.6-luna",
                    error_metadata={"reasoning_signal_present": True,
                                    "reasoning_token_count": 1},
                )}
            def mappings(self, key):
                return ({"custom_id": "d", "stage": "final_decision"},)
            def latest(self, key):
                return {"status": "RESULTS_RETRIEVED", "missing_custom_ids": []}
            def mark_ingested(self, key):
                pass
            def submit(self, key):
                Coordinator.submits += 1
        args = [
            "--provider", "openai", "--model", "gpt_5_6_luna_openai",
            "--reasoning-mode", "on", "--candidate-cap", "128",
            "--run-id", "p", "--poll-existing", "key",
            "--acknowledge-paid-provider-call", "I AUTHORIZE ONE PAID BATCH PREFLIGHT",
        ]
        with patch("scripts.preflight_promise_switch_batch.BatchCoordinator", Coordinator), patch("builtins.print"):
            self.assertEqual(preflight_main(args), 0)
        self.assertEqual(Coordinator.submits, 0)

    def test_production_batch_cli_requires_dual_submit_guard_and_poll_never_submits(self):
        config = configured("openai", "gpt_5_6_luna_openai", "off")
        parameters = type("Parameters", (), {
            "execution_mode": "provider_batch", "model_configs": (config,),
        })()

        class Coordinator:
            submits = 0
            def submit(self, key):
                self.submits += 1
                return "job"

        class Runner:
            instance = None
            def __init__(self, *args, **kwargs):
                self.coordinator = Coordinator()
                Runner.instance = self
            def poll_and_retrieve(self, key):
                return None

        base = ["submit", "--run-id", "r", "--batch-key", "key"]
        patches = (
            patch("scripts.run_promise_switch_batch.parameters_from_manifest", return_value=parameters),
            patch("scripts.run_promise_switch_batch.PromiseSwitchBatchRunner", Runner),
        )
        with patches[0], patches[1], patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "live-submit"):
                batch_cli_main(base)
            with self.assertRaisesRegex(SystemExit, "acknowledge"):
                batch_cli_main([*base, "--live-submit"])
            with self.assertRaisesRegex(SystemExit, "acknowledge"):
                batch_cli_main([*base, "--live-submit", "--acknowledge-paid-provider-call", "wrong"])
            with self.assertRaisesRegex(SystemExit, "live-submit"):
                batch_cli_main([
                    *base, "--acknowledge-paid-provider-call",
                    "I AUTHORIZE PAID PROMISE SWITCH BATCH EXECUTION",
                ])
            self.assertEqual(batch_cli_main([
                *base, "--live-submit", "--acknowledge-paid-provider-call",
                "I AUTHORIZE PAID PROMISE SWITCH BATCH EXECUTION",
            ]), 0)
            self.assertEqual(Runner.instance.coordinator.submits, 1)
            self.assertEqual(batch_cli_main([
                "poll-ingest", "--run-id", "r", "--batch-key", "key",
                "--acknowledge-paid-provider-call",
                "I AUTHORIZE PROMISE SWITCH BATCH RESULT RETRIEVAL",
            ]), 0)
            self.assertEqual(Runner.instance.coordinator.submits, 0)

    def test_semantic_judge_requires_dual_paid_guard(self):
        required = [
            "--sample", "x", "--private-sample-key", "y", "--provider", "nebius",
            "--model", "gpt_oss_120b_nebius", "--reasoning-mode", "on",
            "--temperature", "0", "--max-output-tokens", "1", "--judge-id", "j",
            "--judge-config-id", "c", "--assignment", "all",
            "--annotations-out", "a", "--provenance-out", "p",
        ]
        with self.assertRaisesRegex(SystemExit, "live-submit"):
            semantic_judge_main(required)
        with self.assertRaisesRegex(SystemExit, "acknowledge"):
            semantic_judge_main([*required, "--live-submit"])

    def test_semantic_v2_new_only_reuses_base_and_refuses_overwrite(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            run_dir = work / "future_run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("{}")
            source = run_dir / "message_events.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in [
                {"trial_id": "new-1", "message_recorded": True, "raw_message": "new message", "model": "gpt-5.6-luna"},
                {"trial_id": "excluded", "message_recorded": False, "raw_message": "not eligible"},
            ]))
            output = work / "semantic_v2"
            args = [
                "--source-run", f"GPT-5.6 Luna — no-think={run_dir}",
                "--gpt-luna-routing-policy", "narrow", "--output-dir", str(output),
            ]
            audited = ({"run_id": "future_run"}, {
                "model_id": "gpt-5.6-luna", "model_alias": "gpt_5_6_luna_openai",
                "provider": "openai", "reasoning_mode": "off", "family": "gpt_5_6_luna",
            })
            with patch("scripts.prepare_promise_switch_semantic_v2.audit_future_run", return_value=audited):
                self.assertEqual(prepare_semantic_v2(args), 0)
            manifest = json.loads((output / "corpus_manifest.json").read_text())
            self.assertEqual(manifest["frozen_base"]["n"], 2738)
            self.assertEqual(manifest["new_items"]["n"], 1)
            self.assertEqual(manifest["combined_n"], 2739)
            self.assertEqual(len((output / "new_items_pending_coding.jsonl").read_text().splitlines()), 1)
            self.assertTrue(manifest["frozen_codebook"]["verified"])
            with patch("scripts.prepare_promise_switch_semantic_v2.audit_future_run", return_value=audited), self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                prepare_semantic_v2(args)

    def test_actual_semantic_judge_runner_consumes_v2_pending_items_offline(self):
        class FakeJudgeClient:
            def __init__(self, *args, **kwargs):
                pass
            def complete(self, **kwargs):
                annotation = {
                    "item_id": item_id, "message": "new message",
                    "codebook_version": "promise_switch_message_codebook_v1",
                    "explicit_promise": "no", "commitment_present": "no",
                    "statement_type": "no_substantive_statement",
                    "promised_action": "NOT_APPLICABLE",
                    "conditionality": "not_applicable", "condition_text": "",
                    "refusal_present": "no_refusal", "ambiguous_response": "no",
                }
                return NormalizedResponse(
                    "success", json.dumps(annotation),
                    error_metadata={"reasoning_signal_present": True},
                )
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            run_dir = work / "future-run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("{}")
            source = run_dir / "message_events.jsonl"
            source.write_text(json.dumps({
                "trial_id": "new-1", "run_id": "future-run",
                "message_event_id": "message:new-1", "message_recorded": True,
                "raw_message": "new message", "model": "gpt-5.6-luna",
            }) + "\n")
            corpus = work / "corpus"
            audited = ({"run_id": "future-run"}, {
                "model_id": "gpt-5.6-luna", "model_alias": "gpt_5_6_luna_openai",
                "provider": "openai", "reasoning_mode": "off", "family": "gpt_5_6_luna",
            })
            with patch("scripts.prepare_promise_switch_semantic_v2.audit_future_run", return_value=audited):
                prepare_semantic_v2([
                    "--source-run", f"GPT-5.6 Luna — no-think={run_dir}",
                    "--gpt-luna-routing-policy", "narrow", "--output-dir", str(corpus),
                ])
            pending = [json.loads(line) for line in
                       (corpus / "new_items_pending_coding.jsonl").read_text().splitlines()]
            item_id = pending[0]["item_id"]
            annotations = work / "judge_annotations_v2.jsonl"
            provenance = work / "judge_provenance_v2.json"
            with patch(
                "scripts.run_promise_switch_message_judge.ProviderAgnosticClient",
                FakeJudgeClient,
            ), patch("builtins.print"):
                result = semantic_judge_main([
                    "--sample", str(corpus / "new_items_blinded.csv"),
                    "--private-sample-key", str(corpus / "private_new_items_key.json"),
                    "--semantic-v2-manifest", str(corpus / "corpus_manifest.json"),
                    "--provider", "nebius", "--model", "gpt_oss_120b_nebius",
                    "--reasoning-mode", "on", "--temperature", "0",
                    "--max-output-tokens", "256", "--judge-id", "judge-v2",
                    "--judge-config-id", "judge-config-v2", "--assignment", "all",
                    "--annotations-out", str(annotations),
                    "--provenance-out", str(provenance),
                    "--expected-target-count", "1",
                    "--live-submit", "--acknowledge-paid-provider-call",
                    "I AUTHORIZE PAID PROMISE SWITCH SEMANTIC JUDGE EXECUTION",
                ])
            self.assertEqual(result, 0)
            rows = [json.loads(line) for line in annotations.read_text().splitlines()]
            self.assertEqual([(row["item_id"], row["message"]) for row in rows], [
                (item_id, "new message"),
            ])

    def test_semantic_v2_codebook_hash_mismatch_fails(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            altered = work / "codebook.md"
            altered.write_text("altered")
            source = work / "run"
            source.mkdir()
            with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                prepare_semantic_v2([
                    "--source-run", f"GPT-5.6 Luna — no-think={source}",
                    "--gpt-luna-routing-policy", "conservative",
                    "--codebook", str(altered), "--output-dir", str(work / "out"),
                ])

    def test_semantic_v2_missing_family_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run"
            source.mkdir()
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                prepare_semantic_v2(["--source-run", f"GPT-5.6 Luna — no-think={source}", "--output-dir", str(Path(directory) / "out")])

    def test_semantic_v2_finalize_requires_complete_unique_coding_and_preserves_v1(self):
        root = Path(__file__).resolve().parents[1]
        base = root / "notes/promise_switch/semantic_coding/promise_switch_message_codebook_v1/final_full_semantic_annotations_n2738.jsonl"
        before = hashlib.sha256(base.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            run_dir = work / "future_run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("{}")
            (run_dir / "message_events.jsonl").write_text(json.dumps({
                "trial_id": "new-1", "message_event_id": "message:new-1",
                "message_recorded": True, "raw_message": "new message",
            }) + "\n")
            expected = next(row for row in FUTURE_CONFIGURATIONS
                            if row["label"] == "GPT-5.6 Luna — no-think")
            with patch(
                "scripts.prepare_promise_switch_semantic_v2.audit_future_run",
                return_value=({"run_id": "future_run"}, expected),
            ):
                corpus = work / "prepared"
                prepare_semantic_v2([
                    "--source-run", f"GPT-5.6 Luna — no-think={run_dir}",
                    "--gpt-luna-routing-policy", "narrow", "--output-dir", str(corpus),
                ])
            item = json.loads((corpus / "new_items_pending_coding.jsonl").read_text())
            annotation = {
                "item_id": item["item_id"], "message": "new message",
                "codebook_version": "promise_switch_message_codebook_v1",
                "explicit_promise": "no", "commitment_present": "no",
                "statement_type": "no_substantive_statement",
                "promised_action": "NOT_APPLICABLE", "conditionality": "not_applicable",
                "condition_text": "", "refusal_present": "no_refusal",
                "ambiguous_response": "no", "judge_id": "j", "judge_provider": "nebius",
                "judge_model_id": "m", "judge_reasoning_mode": "on", "judge_config_id": "c",
            }
            empty = work / "empty.jsonl"
            empty.write_text("")
            with patch(
                "scripts.finalize_promise_switch_semantic_v2.audit_future_run",
                return_value=({"run_id": "future_run"}, expected),
            ), self.assertRaisesRegex(SystemExit, "incomplete"):
                finalize_semantic_v2([
                    "--prepared-manifest", str(corpus / "corpus_manifest.json"),
                    "--new-annotations", str(empty), "--output-dir", str(work / "bad-empty"),
                ])
            duplicate = work / "duplicate.jsonl"
            duplicate.write_text((json.dumps(annotation) + "\n") * 2)
            with patch(
                "scripts.finalize_promise_switch_semantic_v2.audit_future_run",
                return_value=({"run_id": "future_run"}, expected),
            ), self.assertRaisesRegex(SystemExit, "duplicate"):
                finalize_semantic_v2([
                    "--prepared-manifest", str(corpus / "corpus_manifest.json"),
                    "--new-annotations", str(duplicate), "--output-dir", str(work / "bad-duplicate"),
                ])
            coded = work / "coded.jsonl"
            coded.write_text(json.dumps(annotation) + "\n")
            final = work / "final"
            with patch(
                "scripts.finalize_promise_switch_semantic_v2.audit_future_run",
                return_value=({"run_id": "future_run"}, expected),
            ):
                self.assertEqual(finalize_semantic_v2([
                    "--prepared-manifest", str(corpus / "corpus_manifest.json"),
                    "--new-annotations", str(coded), "--output-dir", str(final),
                ]), 0)
            activation = json.loads((final / "semantic_v2_activation.json").read_text())
            self.assertEqual((activation["new_item_count"], activation["combined_count"]), (1, 2739))
            self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), before)
            import scripts.analyze_promise_switch_final as analysis_script
            with patch.object(analysis_script, "ANNOTATION_PATH", base), patch.object(
                analysis_script, "SEMANTIC_KEY_PATH",
                root / "notes/promise_switch/semantic_coding/promise_switch_message_codebook_v1/private_full_semantic_key_n2738.json",
            ), patch.object(
                analysis_script, "SEMANTIC_EXPECTED_COUNT", 2738,
            ), patch.object(analysis_script, "SEMANTIC_V2_ACTIVE", False):
                analysis_script.activate_semantic_v2(
                    final / "semantic_v2_activation.json",
                    ((str(run_dir), "label", "family", "off", "OpenAI"),),
                )
                self.assertTrue(analysis_script.SEMANTIC_V2_ACTIVE)
                self.assertEqual(analysis_script.SEMANTIC_EXPECTED_COUNT, 2739)

    def test_validation_supplement_is_audited_blinded_separate_and_exact_10x4(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            args = []
            audited = {}
            for config in FUTURE_CONFIGURATIONS:
                run_dir = work / config["model_alias"] / config["reasoning_mode"]
                run_dir.mkdir(parents=True)
                (run_dir / "manifest.json").write_text("{}")
                (run_dir / "message_events.jsonl").write_text("".join(
                    json.dumps({
                        "trial_id": f"{config['label']}-{index}",
                        "message_event_id": f"event-{index}",
                        "message_recorded": True, "raw_message": f"message {index}",
                    }) + "\n" for index in range(12)
                ))
                args.extend(["--source-run", f"{config['label']}={run_dir}"])
                audited[str(run_dir.resolve())] = ({"run_id": run_dir.name}, config)
            output = work / "sample.json"
            provenance = work / "provenance.json"
            def gate(label, path):
                return audited[str(path.resolve())]
            with patch("scripts.prepare_promise_switch_validation_supplement.audit_future_run", side_effect=gate) as checks:
                self.assertEqual(prepare_validation([
                    *args, "--seed", "42", "--output", str(output),
                    "--provenance-output", str(provenance),
                ]), 0)
            sample = json.loads(output.read_text())
            private = json.loads(provenance.read_text())
            self.assertEqual(len(sample["items"]), 40)
            self.assertEqual(len({row["blind_id"] for row in sample["items"]}), 40)
            self.assertNotIn("sources", sample)
            self.assertNotIn("seed", sample)
            self.assertEqual(len(private["sources"]), 4)
            self.assertEqual(checks.call_count, 4)
            first_run = Path(next(iter(audited)))
            (first_run / "message_events.jsonl").write_text("".join(
                json.dumps({
                    "trial_id": f"short-{index}", "message_event_id": f"short-{index}",
                    "message_recorded": True, "raw_message": f"short {index}",
                }) + "\n" for index in range(9)
            ))
            with patch(
                "scripts.prepare_promise_switch_validation_supplement.audit_future_run",
                side_effect=gate,
            ), self.assertRaisesRegex(SystemExit, "only 9"):
                prepare_validation([
                    *args, "--seed", "42", "--output", str(work / "short.json"),
                    "--provenance-output", str(work / "short-private.json"),
                ])
            with self.assertRaisesRegex(SystemExit, "overwrite"):
                prepare_validation([
                    *args, "--seed", "42", "--output", str(output),
                    "--provenance-output", str(provenance),
                ])


if __name__ == "__main__":
    unittest.main()
