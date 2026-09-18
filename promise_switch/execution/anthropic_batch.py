"""Native Anthropic Message Batches adapter for Promise Switch."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import certifi

from ..credentials import require_api_key
from ..schemas import NormalizedResponse
from .batch_transport import (
    BatchRequestMapping, BatchSubmissionError, ProviderBatchStatus,
)
from .transport import REQUEST_USER_AGENT


class AnthropicBatchProvider:
    name = "anthropic"

    def __init__(
        self, *, base_url: str = "https://api.anthropic.com/v1/",
        api_key_env: str = "ANTHROPIC_API_KEY", timeout_seconds: float = 90.0,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.urlopen = urlopen
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def serialize(self, mapping: BatchRequestMapping) -> dict[str, Any]:
        body = mapping.request.body
        messages = body.get("messages")
        if (
            mapping.request.model_id != "claude-sonnet-5"
            or not isinstance(messages, list) or len(messages) != 2
            or [message.get("role") for message in messages] != ["system", "user"]
            or body.get("temperature") != 1.0
            or not isinstance(body.get("max_tokens"), int)
            or "max_completion_tokens" in body
        ):
            raise ValueError("invalid Claude Sonnet 5 Promise Switch Batch request")
        thinking = body.get("thinking")
        is_off = thinking == {"type": "disabled"} and "output_config" not in body
        is_on = (
            thinking == {"type": "adaptive", "display": "omitted"}
            and body.get("output_config") == {"effort": "xhigh"}
        )
        if not (is_off or is_on):
            raise ValueError("invalid Claude Sonnet 5 reasoning configuration")
        if any(key in body for key in ("tools", "tool_choice", "response_format")):
            raise ValueError("tools and response formatting are forbidden by the frozen protocol")
        params = {key: value for key, value in body.items() if key != "messages"}
        params["system"] = messages[0]["content"]
        params["messages"] = [dict(messages[1])]
        return {"custom_id": mapping.custom_id, "params": params}

    def submit(self, input_path: Path, *, metadata: Mapping[str, str]) -> str:
        del metadata
        # Resolve credentials before entering any operation that could contact the
        # provider, so a missing key is unambiguously fail-closed.
        self._api_key()
        requests = [json.loads(line) for line in input_path.read_text().splitlines() if line]
        try:
            payload = self._json("POST", "/messages/batches", {"requests": requests})
        except urllib.error.HTTPError as exc:
            raise BatchSubmissionError(
                f"Anthropic Message Batch create was rejected with HTTP {exc.code}",
                known_not_submitted=True,
            ) from exc
        except Exception as exc:
            raise BatchSubmissionError(
                f"Anthropic Message Batch create outcome is ambiguous: {type(exc).__name__}",
                known_not_submitted=False,
            ) from exc
        job_id = payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BatchSubmissionError(
                "Anthropic Message Batch create returned no job ID",
                known_not_submitted=False,
            )
        return job_id

    def poll(self, provider_job_id: str) -> ProviderBatchStatus:
        raw = self._json("GET", f"/messages/batches/{provider_job_id}")
        processing = str(raw.get("processing_status", "unknown"))
        terminal = processing == "ended"
        # An ended batch can contain per-request errored/expired/canceled results;
        # those must be retrieved and classified rather than dropping the batch.
        return ProviderBatchStatus(processing, terminal, terminal, raw)

    def retrieve(
        self, provider_job_id: str, status: ProviderBatchStatus
    ) -> Sequence[dict[str, Any]]:
        del status
        raw = self._request("GET", f"/messages/batches/{provider_job_id}/results", None)
        return [
            _strip_hidden_reasoning(json.loads(line))
            for line in raw.decode("utf-8").splitlines() if line.strip()
        ]

    def normalize(self, result: Mapping[str, Any]) -> NormalizedResponse:
        wrapped = result.get("result")
        if not isinstance(wrapped, Mapping):
            return NormalizedResponse(
                status="malformed_api_response", raw_response="",
                error_metadata={"type": "missing_batch_result"},
            )
        result_type = str(wrapped.get("type", ""))
        telemetry = result.get("_promise_switch_reasoning", {})
        if result_type != "succeeded":
            error = wrapped.get("error", {})
            text = json.dumps(error).lower()
            return NormalizedResponse(
                status=("timeout" if "timeout" in text else result_type
                        if result_type in {"expired", "canceled"} else "error"),
                raw_response="",
                http_status=_error_status(error),
                error_metadata={"provider_result_type": result_type, "provider_error": error},
            )
        message = wrapped.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
            return NormalizedResponse(
                status="malformed_api_response", raw_response="",
                error_metadata={"type": "missing_message_content"},
            )
        content = "".join(
            str(block.get("text", "")) for block in message["content"]
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        stop = str(message.get("stop_reason", "")) or None
        finish = {
            "end_turn": "stop", "stop_sequence": "stop", "max_tokens": "max_tokens",
            "model_context_window_exceeded": "max_tokens", "refusal": "refusal",
        }.get(stop, stop)
        return NormalizedResponse(
            status="success", raw_response=content,
            returned_model_id=(str(message.get("model")) if message.get("model") else None),
            provider_response_id=str(message.get("id")) if message.get("id") else None,
            http_status=200, finish_reason=finish,
            usage=dict(message.get("usage", {})) if isinstance(message.get("usage"), Mapping) else {},
            error_metadata={
                "provider_refusal": stop == "refusal",
                "thinking_present": bool(telemetry.get("thinking_present")),
                "thinking_token_count": telemetry.get("thinking_token_count"),
            },
        )

    def _api_key(self) -> str:
        return require_api_key(self.api_key_env)

    def _json(
        self, method: str, path: str, body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        raw = self._request(
            method, path, json.dumps(body).encode() if body is not None else None
        )
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("provider returned a non-object JSON envelope")
        return value

    def _request(self, method: str, path: str, data: bytes | None) -> bytes:
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={
                "x-api-key": self._api_key(), "anthropic-version": "2023-06-01",
                "content-type": "application/json", "User-Agent": REQUEST_USER_AGENT,
            },
        )
        with self.urlopen(
            request, timeout=self.timeout_seconds, context=self.ssl_context
        ) as response:
            return response.read()


def _strip_hidden_reasoning(row: dict[str, Any]) -> dict[str, Any]:
    """Remove thinking/signature blocks before any result artifact is persisted."""
    wrapped = row.get("result")
    message = wrapped.get("message") if isinstance(wrapped, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return row
    thinking_present = False
    visible = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text":
            visible.append(dict(block))
        else:
            thinking_present = True
    usage = message.get("usage", {}) if isinstance(message.get("usage"), Mapping) else {}
    details = usage.get("output_tokens_details", {})
    count = details.get("thinking_tokens") if isinstance(details, Mapping) else None
    message["content"] = visible
    row["_promise_switch_reasoning"] = {
        "thinking_present": thinking_present or bool(count),
        "thinking_token_count": count,
    }
    return row


def _error_status(error: Any) -> int | None:
    if isinstance(error, Mapping):
        try:
            return int(error.get("status_code"))
        except (TypeError, ValueError):
            return None
    return None
