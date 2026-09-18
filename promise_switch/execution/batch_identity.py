"""Fail-closed production identity checks for native Promise Switch Batch attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..schemas import ModelConfig, NormalizedResponse


@dataclass(frozen=True)
class BatchIdentityCheck:
    failures: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return not self.failures

    @property
    def failure_cause(self) -> str | None:
        if any(item.startswith("returned_model_id") for item in self.failures):
            return "model_identity_failure"
        return "request_configuration_identity_failure" if self.failures else None


def verify_production_batch_attempt(
    *, config: ModelConfig, request: Mapping[str, Any],
    response: NormalizedResponse,
) -> BatchIdentityCheck:
    """Compare a durable request/response with its immutable Claude manifest config."""
    if config.provider != "anthropic" or config.model_id != "claude-sonnet-5":
        return BatchIdentityCheck(())

    failures: list[str] = []
    body = request.get("body")
    metadata = request.get("metadata")
    if not isinstance(body, Mapping):
        return BatchIdentityCheck(("request.body missing",))
    if not isinstance(metadata, Mapping):
        failures.append("request.metadata missing")
        metadata = {}

    expected_cap = config.production_max_output_tokens
    expected_kwargs = config.provider_request_kwargs
    checks = (
        (config.batch_protocol == "anthropic_message_batches", "batch_protocol mismatch"),
        (config.output_token_field == "max_tokens", "output_token_field mismatch"),
        (request.get("model_id") == config.model_id, "request.model_id mismatch"),
        (body.get("model") == config.model_id, "body.model mismatch"),
        (metadata.get("provider") == config.provider, "metadata.provider mismatch"),
        (metadata.get("model_alias") == config.model_alias, "metadata.model_alias mismatch"),
        (metadata.get("reasoning_mode") == config.reasoning_mode,
         "metadata.reasoning_mode mismatch"),
        (body.get("temperature") == 1.0, "temperature mismatch"),
        (isinstance(expected_cap, int) and body.get("max_tokens") == expected_cap,
         "max_tokens mismatch"),
        ("max_completion_tokens" not in body, "wrong token field"),
        (body.get("thinking") == expected_kwargs.get("thinking"), "thinking mismatch"),
        (body.get("output_config") == expected_kwargs.get("output_config"),
         "output_config mismatch"),
    )
    failures.extend(message for valid, message in checks if not valid)

    if response.status == "success":
        if not response.returned_model_id:
            failures.append("returned_model_id missing")
        elif response.returned_model_id != config.model_id:
            failures.append(
                f"returned_model_id mismatch: expected {config.model_id}, "
                f"received {response.returned_model_id}"
            )
        if config.reasoning_mode == "off":
            signal = response.error_metadata.get("thinking_present")
            count = response.error_metadata.get("thinking_token_count")
            if signal is True or isinstance(count, (int, float)) and count > 0:
                failures.append("disabled thinking contradicted by response telemetry")

    return BatchIdentityCheck(tuple(failures))


def verify_claude_wire_configuration(
    *, config: ModelConfig, params: Mapping[str, Any],
) -> BatchIdentityCheck:
    """Verify the exact Anthropic params object persisted for provider submission."""
    if config.provider != "anthropic" or config.model_id != "claude-sonnet-5":
        return BatchIdentityCheck(())
    expected_kwargs = config.provider_request_kwargs
    expected_cap = config.production_max_output_tokens
    checks = (
        (params.get("model") == config.model_id, "wire model mismatch"),
        (params.get("temperature") == 1.0, "wire temperature mismatch"),
        (isinstance(expected_cap, int) and params.get("max_tokens") == expected_cap,
         "wire max_tokens mismatch"),
        ("max_completion_tokens" not in params, "wire token field mismatch"),
        (params.get("thinking") == expected_kwargs.get("thinking"),
         "wire thinking mismatch"),
        (params.get("output_config") == expected_kwargs.get("output_config"),
         "wire output_config mismatch"),
        (isinstance(params.get("system"), str), "wire system prompt missing"),
        (isinstance(params.get("messages"), list), "wire messages missing"),
    )
    return BatchIdentityCheck(tuple(message for valid, message in checks if not valid))
