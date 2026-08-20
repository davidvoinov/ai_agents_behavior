"""Exactly-one-request technical validation for a final model configuration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .client import CompletionTransport, ProviderAgnosticClient
from .design import Condition, treatment_metadata
from .execution.attrition import classify_transport
from .parser import parse_decision
from .prompts import final_decision_messages
from .schemas import ModelConfig, NormalizedResponse, sanitize_for_serialization
from .version import PROMPT_VERSION, prompt_templates_hash

PREFLIGHT_SCHEMA_VERSION = "promise_switch_exact_final_preflight_v1"
PREFLIGHT_DATASET_ROLE = "technical_preflight"


@dataclass(frozen=True)
class PreflightParameters:
    preflight_id: str
    output_dir: Path
    model_config: ModelConfig
    temperature: float = 1.0

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / self.preflight_id

    def __post_init__(self) -> None:
        if (
            not self.preflight_id.strip()
            or "/" in self.preflight_id
            or self.preflight_id in {".", ".."}
        ):
            raise ValueError("preflight_id must be a non-empty path-safe name")
        if self.temperature != 1.0:
            raise ValueError("exact-final preflight requires frozen temperature 1.0")


class ExactFinalPreflight:
    """Make one request, evaluate it, and write one non-scientific artifact."""

    def __init__(
        self,
        parameters: PreflightParameters,
        *,
        transport: CompletionTransport,
        timestamp: Callable[[], str] | None = None,
        git_head: Callable[[], str] | None = None,
    ) -> None:
        self.parameters = parameters
        self.transport = transport
        self.timestamp = timestamp or _utc_now
        self.git_head = git_head or _git_head

    def run(self) -> dict[str, Any]:
        artifact_path = self.parameters.artifact_dir / "preflight.json"
        if artifact_path.exists():
            raise FileExistsError(f"preflight artifact already exists: {artifact_path}")
        config = self.parameters.model_config
        metadata = treatment_metadata(Condition.NO_COMMUNICATION, self.parameters.preflight_id)
        messages = final_decision_messages(
            condition=Condition.NO_COMMUNICATION,
            original_partner_id=metadata.original_partner_id,
            final_partner_id=metadata.final_partner_id,
        )
        client = ProviderAgnosticClient(config, transport=self.transport)
        request = client.build_request(
            messages=messages,
            temperature=self.parameters.temperature,
            max_tokens=config.production_max_output_tokens,
        )
        request = replace(
            request,
            metadata={**request.metadata, "stage": "final_decision", "dataset_role": PREFLIGHT_DATASET_ROLE},
        )
        request_count = 0
        try:
            request_count = 1
            response = self.transport.complete(request)
            artifact = self._evaluate(request.body, response, request_count)
        except Exception as exc:  # one attempted call; never retry an unexpected transport fault
            artifact = self._base_artifact(request.body, request_count)
            artifact.update({
                "result": "FAIL",
                "failure_category": "unexpected_transport_exception",
                "http_status": None,
                "provider_failure_category": "transport_or_connection_error",
                "safe_error_metadata": {},
                "exception_type": type(exc).__name__,
                "finish_reason": None,
                "truncated": False,
                "parser_success": False,
                "mechanically_parsed_action": None,
                "reasoning_verification": "NOT_EVALUATED",
                "reasoning_present": None,
                "safe_usage": {},
                "provider_response_id": None,
                "provider_request_id": None,
            })
        safe_artifact = sanitize_for_serialization(artifact)
        _write_exclusive_json(artifact_path, safe_artifact)
        return safe_artifact

    def _evaluate(
        self, request_body: dict[str, Any], response: NormalizedResponse, request_count: int,
    ) -> dict[str, Any]:
        artifact = self._base_artifact(request_body, request_count)
        transport_outcome = classify_transport(response)
        parsed = parse_decision(response.raw_response) if response.status == "success" else None
        truncated = response.finish_reason in {"length", "max_tokens"}
        response_succeeded = response.status == "success"
        reasoning_present = _reasoning_present(response) if response_succeeded else None
        reasoning_verification = (
            _reasoning_verification(self.parameters.model_config.reasoning_mode, reasoning_present)
            if response_succeeded else "NOT_EVALUATED"
        )
        normal_finish = response.status == "success" and response.finish_reason == "stop"
        parser_success = bool(parsed and parsed.valid)
        failure_category: str | None = None
        if transport_outcome.category != "success":
            failure_category = str(transport_outcome.cause or "provider_failure")
        elif not normal_finish:
            failure_category = "token_truncation" if truncated else "abnormal_finish"
        elif not parser_success:
            failure_category = "invalid_decision_json"
        elif reasoning_verification in {"REASONING_STATE_UNVERIFIABLE", "REASONING_STATE_MISMATCH"}:
            failure_category = reasoning_verification.lower()
        passed = failure_category is None
        artifact.update({
            "result": "PASS" if passed else "FAIL",
            "failure_category": failure_category,
            "http_status": response.http_status,
            "provider_failure_category": _provider_failure_category(response),
            "safe_error_metadata": _safe_error_metadata(response),
            "finish_reason": response.finish_reason,
            "truncated": truncated,
            "parser_success": parser_success,
            "mechanically_parsed_action": parsed.decision if parser_success and parsed else None,
            "reasoning_verification": reasoning_verification,
            "reasoning_present": reasoning_present,
            "safe_usage": _safe_usage(response.usage),
            "provider_response_id": response.provider_response_id,
            "provider_request_id": response.provider_request_id,
        })
        return artifact

    def _base_artifact(self, request_body: dict[str, Any], request_count: int) -> dict[str, Any]:
        config = self.parameters.model_config
        safe_request_configuration = {
            "model": request_body["model"],
            "temperature": request_body["temperature"],
            "max_tokens": request_body["max_tokens"],
            "provider_request_kwargs": config.provider_request_kwargs,
        }
        canonical_body = json.dumps(request_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
            "dataset_role": PREFLIGHT_DATASET_ROLE,
            "preflight_id": self.parameters.preflight_id,
            "timestamp_utc": self.timestamp(),
            "git_head": self.git_head(),
            "provider": config.provider,
            "endpoint": config.base_url,
            "exact_model_id": config.model_id,
            "model_alias": config.model_alias,
            "reasoning_mode": config.reasoning_mode,
            "reasoning_parameters": config.provider_request_kwargs,
            "requested_temperature": self.parameters.temperature,
            "configured_output_cap": config.production_max_output_tokens,
            "prompt_version": PROMPT_VERSION,
            "prompt_hash": prompt_templates_hash(),
            "request_count": request_count,
            "request_configuration": safe_request_configuration,
            "request_body_sha256": hashlib.sha256(canonical_body.encode("utf-8")).hexdigest(),
        }


def _reasoning_present(response: NormalizedResponse) -> bool | None:
    metadata = response.error_metadata
    explicit = [
        metadata.get("reasoning_signal_present"),
        metadata.get("reasoning_trace_present"),
    ]
    if any(value is True for value in explicit):
        return True
    reasoning_tokens = _reasoning_tokens(response.usage)
    if reasoning_tokens is not None:
        return reasoning_tokens > 0
    if any(value is False for value in explicit):
        return False
    return None


def _reasoning_verification(mode: str, present: bool | None) -> str:
    if mode == "unavailable":
        return "NOT_APPLICABLE"
    if present is None:
        return "REASONING_STATE_UNVERIFIABLE"
    if mode == "on":
        return "VERIFIED_ON" if present else "REASONING_STATE_UNVERIFIABLE"
    return "REASONING_STATE_MISMATCH" if present else "VERIFIED_OFF"


def _reasoning_tokens(value: Any) -> int | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "reasoning" in str(key).lower() and isinstance(item, (int, float)):
                return int(item)
            nested = _reasoning_tokens(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _reasoning_tokens(item)
            if nested is not None:
                return nested
    return None


def _safe_usage(usage: Any) -> dict[str, Any]:
    return sanitize_for_serialization(usage if isinstance(usage, dict) else {})


def _provider_failure_category(response: NormalizedResponse) -> str | None:
    """Return a deterministic safe diagnosis without parsing provider error text."""
    if response.status == "success":
        return None
    by_status = {
        401: "authentication_error",
        402: "billing_or_quota_payment_required",
        403: "permission_forbidden",
        404: "model_or_endpoint_not_found",
        429: "rate_limit",
    }
    if response.http_status in by_status:
        return by_status[response.http_status]
    if response.http_status is not None and 500 <= response.http_status <= 599:
        return "provider_server_error"
    if response.status == "malformed_api_response":
        return "malformed_api_response"
    if response.http_status is None:
        return "transport_or_connection_error"
    return "provider_http_error"


_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)(authorization\s*:|bearer\s+\S+|api[_ -]?key\s*[:=]\s*\S+|"
    r"cookie\s*:|sk-[a-z0-9_-]{8,})"
)


def _safe_error_metadata(response: NormalizedResponse) -> dict[str, Any]:
    """Whitelist bounded diagnostic scalars; never persist arbitrary error payloads."""
    metadata = response.error_metadata if isinstance(response.error_metadata, dict) else {}
    provider_error = metadata.get("provider_error")
    source = provider_error if isinstance(provider_error, dict) else metadata
    safe: dict[str, Any] = {}
    for source_key, output_key in (
        ("type", "provider_error_type"),
        ("code", "provider_error_code"),
        ("status", "provider_error_status"),
    ):
        value = _safe_diagnostic_scalar(source.get(source_key))
        if value is not None:
            safe[output_key] = value
    if response.provider_request_id:
        request_id = _safe_diagnostic_scalar(response.provider_request_id)
        if request_id is not None:
            safe["provider_request_id"] = request_id
    return safe


def _safe_diagnostic_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    bounded = value.strip()[:160]
    if not bounded:
        return None
    if _SENSITIVE_VALUE_PATTERN.search(bounded):
        return "[REDACTED]"
    return bounded


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
