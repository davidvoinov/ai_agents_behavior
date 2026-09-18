from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from promise_switch import credentials
from promise_switch.client import ProviderAgnosticClient, RequestSpec
from promise_switch.execution.anthropic_batch import AnthropicBatchProvider
from promise_switch.execution.batch_transport import make_mapping
from promise_switch.execution.openai_batch import OpenAIBatchProvider
from promise_switch.execution.transport import OpenAICompatibleTransport
from promise_switch.providers import create_model_config


class _Response:
    status = 200
    headers: dict[str, str] = {}

    def __init__(self, value: bytes):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self) -> bytes:
        return self.value


class PromiseSwitchCredentialTests(unittest.TestCase):
    def dotenv(self, value: str) -> tuple[tempfile.TemporaryDirectory, Path]:
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / ".env"
        path.write_text(value, encoding="utf-8")
        return directory, path

    def test_key_exists_only_in_root_dotenv(self):
        directory, path = self.dotenv("PROMISE_SWITCH_TEST_KEY=dotenv-secret\n")
        self.addCleanup(directory.cleanup)
        with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
            os.environ, {}, clear=True,
        ):
            self.assertEqual(
                credentials.require_api_key("PROMISE_SWITCH_TEST_KEY"), "dotenv-secret"
            )

    def test_shell_environment_wins_over_dotenv(self):
        directory, path = self.dotenv("PROMISE_SWITCH_TEST_KEY=dotenv-secret\n")
        self.addCleanup(directory.cleanup)
        with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
            os.environ, {"PROMISE_SWITCH_TEST_KEY": "shell-secret"}, clear=True,
        ):
            self.assertEqual(
                credentials.require_api_key("PROMISE_SWITCH_TEST_KEY"), "shell-secret"
            )

    def test_missing_and_blank_keys_fail_closed(self):
        for contents in ("", "PROMISE_SWITCH_TEST_KEY=\n"):
            with self.subTest(contents=contents):
                directory, path = self.dotenv(contents)
                try:
                    with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
                        os.environ, {}, clear=True,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "^PROMISE_SWITCH_TEST_KEY is not set$"
                        ):
                            credentials.require_api_key("PROMISE_SWITCH_TEST_KEY")
                finally:
                    directory.cleanup()

    def test_repository_dotenv_resolution_does_not_depend_on_working_directory(self):
        expected = Path(credentials.__file__).resolve().parents[1] / ".env"
        self.assertEqual(credentials.ROOT_DOTENV_PATH, expected)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            credentials, "load_dotenv"
        ) as loader:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                credentials.load_repository_dotenv()
            finally:
                os.chdir(previous)
        loader.assert_called_once_with(dotenv_path=expected, override=False)

    def test_openai_and_anthropic_batch_paths_use_dotenv_and_never_call_network_when_missing(self):
        for provider, env_name in (
            (OpenAIBatchProvider(), "OPENAI_API_KEY"),
            (AnthropicBatchProvider(), "ANTHROPIC_API_KEY"),
        ):
            with self.subTest(provider=provider.name):
                directory, path = self.dotenv(f"{env_name}=dotenv-secret\n")
                try:
                    with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
                        os.environ, {}, clear=True,
                    ):
                        self.assertEqual(provider._api_key(), "dotenv-secret")
                    path.write_text(f"{env_name}=\n", encoding="utf-8")
                    network = Mock(side_effect=AssertionError("network must not be called"))
                    provider.urlopen = network
                    input_path = Path(directory.name) / "input.jsonl"
                    input_path.write_text("{}\n", encoding="utf-8")
                    with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
                        os.environ, {}, clear=True,
                    ):
                        with self.assertRaisesRegex(RuntimeError, f"^{env_name} is not set$"):
                            provider.submit(input_path, metadata={})
                    network.assert_not_called()
                finally:
                    directory.cleanup()

    def test_semantic_judge_transport_uses_same_dotenv_loader(self):
        directory, path = self.dotenv("JUDGE_API_KEY=dotenv-secret\n")
        self.addCleanup(directory.cleanup)
        response = _Response(json.dumps({
            "id": "mock", "choices": [{
                "message": {"content": "{}"}, "finish_reason": "stop",
            }], "usage": {},
        }).encode())
        with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
            os.environ, {}, clear=True,
        ), patch("promise_switch.execution.transport.urllib.request.urlopen", return_value=response) as call:
            result = OpenAICompatibleTransport().complete(RequestSpec(
                base_url="https://invalid.example", model_id="mock",
                api_key_env="JUDGE_API_KEY", body={}, metadata={},
            ))
        self.assertEqual(result.status, "success")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[0].get_header("Authorization"), "Bearer dotenv-secret")

    def test_loaded_secrets_are_absent_from_serialized_artifacts_and_logs(self):
        directory, path = self.dotenv(
            "OPENAI_API_KEY=openai-serialization-secret\n"
            "ANTHROPIC_API_KEY=anthropic-serialization-secret\n"
        )
        self.addCleanup(directory.cleanup)
        with patch.object(credentials, "ROOT_DOTENV_PATH", path), patch.dict(
            os.environ, {}, clear=True,
        ):
            rows = []
            for provider_name, alias, env_name, provider in (
                ("openai", "gpt_5_6_luna_openai", "OPENAI_API_KEY", OpenAIBatchProvider()),
                (
                    "anthropic", "claude_sonnet_5_anthropic", "ANTHROPIC_API_KEY",
                    AnthropicBatchProvider(),
                ),
            ):
                config = create_model_config(
                    provider=provider_name, model_alias=alias, reasoning_mode="off"
                )
                request = ProviderAgnosticClient(config).build_request(
                    messages=[
                        {"role": "system", "content": "offline"},
                        {"role": "user", "content": "offline"},
                    ],
                    temperature=1.0, max_tokens=256,
                )
                request = RequestSpec(
                    base_url=request.base_url, model_id=request.model_id,
                    api_key_env=request.api_key_env, body=request.body,
                    metadata={**request.metadata, "trial_id": "offline", "stage": "final_decision"},
                )
                mapping = make_mapping(request, 1)
                rows.append({
                    "config": config.to_dict(), "request_log": request.to_log_dict(),
                    "mapping": mapping.to_dict(), "batch_input": provider.serialize(mapping),
                })
                self.assertTrue(credentials.require_api_key(env_name))
            serialized = json.dumps(rows)
        self.assertNotIn("openai-serialization-secret", serialized)
        self.assertNotIn("anthropic-serialization-secret", serialized)
        self.assertIn("OPENAI_API_KEY", serialized)
        self.assertIn("ANTHROPIC_API_KEY", serialized)


if __name__ == "__main__":
    unittest.main()
