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
from promise_switch.execution.attrition import classify_transport
from promise_switch.execution.lock import RunLockError, RunWriterLock
from promise_switch.execution.planner import plan_trials
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
from promise_switch.execution.transport import normalize_reasoning_content
from promise_switch.providers import create_model_config
from promise_switch.schemas import NormalizedResponse
from promise_switch.version import prompt_templates_hash


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
        return ExecutionParameters(
            run_id=run_id, output_dir=self.output, n_per_condition=n, seed=seed,
            model_configs=kwargs.pop("model_configs", (self.config,)),
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
        self.runner(params, transport).run()
        first = self.rows(params, "trials.jsonl")[0]
        self.assertEqual(first["failure_category"], "behavioral")
        attempts = [r for r in self.rows(params, "api_attempts.jsonl")
                    if r["trial_id"] == first["trial_id"]]
        self.assertEqual(len(attempts), 1)

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
        params = self.params(
            dry_run=True, message_max_tokens=4096, decision_max_tokens=4096,
        )
        PromiseSwitchRunner(params).run()
        manifest = RunStorage(params.run_dir).read_manifest()
        self.assertEqual(manifest["max_output_tokens"], 4096)
        self.assertEqual(manifest["token_limits"]["message"], 4096)
        changed = ExecutionParameters(**{
            **params.__dict__, "message_max_tokens": 2048,
            "decision_max_tokens": 2048,
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

    @staticmethod
    def _journal_snapshot(params):
        return {
            name: (params.run_dir / name).read_bytes()
            for name in FILES if (params.run_dir / name).exists()
        }


if __name__ == "__main__":
    unittest.main()
