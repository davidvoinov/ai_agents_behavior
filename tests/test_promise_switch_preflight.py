from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from promise_switch.artifacts import validate_run_directory
from promise_switch.client import RequestSpec
from promise_switch.preflight import ExactFinalPreflight, PreflightParameters
from promise_switch.providers import create_model_config
from promise_switch.schemas import NormalizedResponse
from promise_switch.version import PROMPT_VERSION, prompt_templates_hash


def success(*, reasoning: bool | None = None, raw: str | None = None) -> NormalizedResponse:
    metadata = {} if reasoning is None else {"reasoning_signal_present": reasoning}
    return NormalizedResponse(
        status="success",
        raw_response=raw or '{"decision":"COOPERATE","justification":"Technical check."}',
        provider_response_id="safe-response-id",
        provider_request_id="safe-request-id",
        http_status=200,
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        error_metadata=metadata,
    )


class FakeTransport:
    def __init__(self, response: NormalizedResponse | Exception):
        self.response = response
        self.requests: list[RequestSpec] = []

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ExactFinalPreflightTests(unittest.TestCase):
    def run_preflight(
        self,
        response: NormalizedResponse | Exception,
        *,
        provider: str = "nebius",
        alias: str = "qwen3_32b",
        mode: str = "off",
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = create_model_config(
            provider=provider, model_alias=alias, reasoning_mode=mode,  # type: ignore[arg-type]
        )
        transport = FakeTransport(response)
        parameters = PreflightParameters(
            preflight_id="exact-final-test",
            output_dir=Path(temporary.name),
            model_config=config,
        )
        artifact = ExactFinalPreflight(
            parameters,
            transport=transport,
            timestamp=lambda: "2026-08-20T00:00:00+00:00",
            git_head=lambda: "a" * 40,
        ).run()
        return artifact, transport, parameters.artifact_dir

    def test_success_makes_exactly_one_request(self):
        artifact, transport, _ = self.run_preflight(success(reasoning=False))
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(artifact["request_count"], 1)
        self.assertEqual(artifact["result"], "PASS")

    def test_failure_and_unexpected_exception_make_no_second_request(self):
        for response in (
            NormalizedResponse(status="timeout", raw_response=""),
            RuntimeError("do not serialize this message"),
        ):
            with self.subTest(response=type(response).__name__):
                artifact, transport, _ = self.run_preflight(response)
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(artifact["request_count"], 1)
                self.assertEqual(artifact["result"], "FAIL")

    def test_http_failures_never_retry(self):
        for status in (401, 402, 403, 429, 500):
            with self.subTest(status=status):
                artifact, transport, _ = self.run_preflight(NormalizedResponse(
                    status="error", raw_response="provider failure", http_status=status,
                ))
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(artifact["result"], "FAIL")

    def test_malformed_json_and_invalid_action_fail(self):
        for raw in ("not json", '{"decision":"WAIT","justification":"Technical check."}'):
            with self.subTest(raw=raw):
                artifact, transport, _ = self.run_preflight(
                    success(reasoning=False, raw=raw),
                )
                self.assertEqual(len(transport.requests), 1)
                self.assertFalse(artifact["parser_success"])
                self.assertEqual(artifact["failure_category"], "invalid_decision_json")

    def test_truncation_fails_without_retry(self):
        response = success(reasoning=False)
        response = NormalizedResponse(**{**response.__dict__, "finish_reason": "length"})
        artifact, transport, _ = self.run_preflight(response)
        self.assertEqual(len(transport.requests), 1)
        self.assertTrue(artifact["truncated"])
        self.assertEqual(artifact["failure_category"], "token_truncation")

    def test_exact_production_resolution_and_qwen_serialization(self):
        artifact, transport, _ = self.run_preflight(success(reasoning=True), mode="on")
        request = transport.requests[0]
        self.assertEqual(request.base_url, "https://api.studio.nebius.com/v1/")
        self.assertEqual(request.model_id, "Qwen/Qwen3-32B")
        self.assertEqual(request.body["temperature"], 1.0)
        self.assertEqual(request.body["max_tokens"], 4096)
        self.assertEqual(request.body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertEqual(request.metadata["dataset_role"], "technical_preflight")
        self.assertEqual(artifact["exact_model_id"], request.model_id)

    def test_family_specific_reasoning_serialization(self):
        cases = (
            ("nebius_uk", "glm_5_2_nebius_thinking_off", "off", "chat_template_kwargs", {"enable_thinking": False}),
            ("nebius", "minimax_m3_nebius_thinking_on", "on", "chat_template_kwargs", {"thinking_mode": "enabled"}),
            ("nebius", "deepseek_v4_flash_nebius_thinking_off", "off", "thinking", {"type": "disabled"}),
            ("groq", "gpt_oss_120b_groq", "on", "reasoning_effort", "low"),
        )
        for provider, alias, mode, key, expected in cases:
            with self.subTest(alias=alias):
                signal = mode == "on"
                artifact, transport, _ = self.run_preflight(
                    success(reasoning=signal), provider=provider, alias=alias, mode=mode,
                )
                self.assertEqual(transport.requests[0].body[key], expected)
                self.assertEqual(artifact["reasoning_parameters"][key], expected)

    def test_prompt_identity_and_final_schema_are_frozen(self):
        artifact, transport, _ = self.run_preflight(success(reasoning=False))
        messages = transport.requests[0].body["messages"]
        self.assertEqual(artifact["prompt_version"], PROMPT_VERSION)
        self.assertEqual(artifact["prompt_hash"], prompt_templates_hash())
        self.assertIn('"decision": "COOPERATE or DEFECT"', messages[-1]["content"])
        self.assertNotIn("H8", json.dumps(messages))

    def test_safe_artifact_is_durable_and_contains_no_credentials_or_raw_response(self):
        artifact, _, artifact_dir = self.run_preflight(success(reasoning=False))
        path = artifact_dir / "preflight.json"
        self.assertTrue(path.is_file())
        stored = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(stored).lower()
        self.assertEqual(stored, artifact)
        self.assertNotIn("authorization", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("raw_response", serialized)
        self.assertEqual(stored["dataset_role"], "technical_preflight")

    def test_reasoning_on_off_and_ambiguous_policies(self):
        on, _, _ = self.run_preflight(success(reasoning=True), mode="on")
        off, _, _ = self.run_preflight(success(reasoning=False), mode="off")
        ambiguous, _, _ = self.run_preflight(success(reasoning=None), mode="on")
        mismatch, _, _ = self.run_preflight(success(reasoning=True), mode="off")
        self.assertEqual((on["result"], on["reasoning_verification"]), ("PASS", "VERIFIED_ON"))
        self.assertEqual((off["result"], off["reasoning_verification"]), ("PASS", "VERIFIED_OFF"))
        self.assertEqual(ambiguous["reasoning_verification"], "REASONING_STATE_UNVERIFIABLE")
        self.assertEqual(ambiguous["result"], "FAIL")
        self.assertEqual(mismatch["reasoning_verification"], "REASONING_STATE_MISMATCH")
        self.assertEqual(mismatch["result"], "FAIL")

    def test_reasoning_unavailable_is_not_applicable(self):
        artifact, _, _ = self.run_preflight(
            success(reasoning=None), alias="kimi_k3_nebius", mode="unavailable",
        )
        self.assertEqual(artifact["result"], "PASS")
        self.assertEqual(artifact["reasoning_verification"], "NOT_APPLICABLE")

    def test_reasoning_usage_is_safe_signal(self):
        response = NormalizedResponse(
            **{**success(reasoning=None).__dict__,
               "usage": {"completion_tokens_details": {"reasoning_tokens": 7}}}
        )
        artifact, _, _ = self.run_preflight(response, mode="on")
        self.assertTrue(artifact["reasoning_present"])
        self.assertEqual(artifact["reasoning_verification"], "VERIFIED_ON")

    def test_preflight_artifact_cannot_enter_scientific_analysis(self):
        _, _, artifact_dir = self.run_preflight(success(reasoning=False))
        with self.assertRaisesRegex(ValueError, "not a complete"):
            validate_run_directory(artifact_dir, required_dataset_role="confirmatory")

    def test_no_scientific_run_or_top_up_files_are_created(self):
        _, _, artifact_dir = self.run_preflight(success(reasoning=False))
        self.assertEqual({path.name for path in artifact_dir.iterdir()}, {"preflight.json"})
        for forbidden in ("manifest.json", "trials.jsonl", "planned_trials.jsonl", "top_up_events.jsonl"):
            self.assertFalse((artifact_dir / forbidden).exists())

    def test_artifact_is_immutable_by_preflight_id(self):
        artifact, transport, artifact_dir = self.run_preflight(success(reasoning=False))
        parameters = PreflightParameters(
            preflight_id=artifact["preflight_id"],
            output_dir=artifact_dir.parent,
            model_config=create_model_config(provider="nebius", model_alias="qwen3_32b", reasoning_mode="off"),
        )
        with self.assertRaises(FileExistsError):
            ExactFinalPreflight(parameters, transport=transport).run()
        self.assertEqual(len(transport.requests), 1)


if __name__ == "__main__":
    unittest.main()
