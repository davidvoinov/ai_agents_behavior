"""OpenAI-compatible transport; tests use fake transports, never this network path."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import replace
from typing import Any

import certifi

from ..client import CompletionTransport, RequestSpec
from ..credentials import api_key_from_environment
from ..schemas import NormalizedResponse

# Some providers front their API with edge bot-protection (e.g. Cloudflare) that
# blocks the default urllib User-Agent. A static, identifiable client string
# avoids that without impersonating a browser.
REQUEST_USER_AGENT = "promise-switch-research/1.0"


class OpenAICompatibleTransport(CompletionTransport):
    def __init__(
        self, *, timeout_seconds: float = 90.0,
        ssl_context: ssl.SSLContext | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        # Use an explicit, maintained CA bundle so the transport is reliable in
        # Python installations whose platform trust store is not initialized.
        self.ssl_context = ssl_context or ssl.create_default_context(cafile=certifi.where())

    def complete(self, request: RequestSpec) -> NormalizedResponse:
        api_key = api_key_from_environment(request.api_key_env)
        if not api_key:
            return NormalizedResponse(
                status="error", raw_response="", http_status=401,
                error_metadata={"type": "missing_api_key", "api_key_env": request.api_key_env},
            )
        url = request.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": REQUEST_USER_AGENT,
        }
        http_request = urllib.request.Request(
            url, data=json.dumps(request.body).encode("utf-8"), headers=headers, method="POST"
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(
                http_request, timeout=self.timeout_seconds, context=self.ssl_context,
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
                normalized = normalize_openai_response(
                    raw=raw, http_status=response.status,
                    headers=dict(response.headers.items()),
                    latency_seconds=time.monotonic() - started,
                )
                cleaned, trace_metadata = normalize_reasoning_content(
                    normalized.raw_response,
                    provider=str(request.metadata.get("provider", "")),
                    model_id=request.model_id,
                    reasoning_mode=str(request.metadata.get("reasoning_mode", "")),
                )
                if trace_metadata:
                    normalized = replace(
                        normalized,
                        raw_response=cleaned,
                        error_metadata={**normalized.error_metadata, **trace_metadata},
                    )
                return normalized
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return NormalizedResponse(
                status="error", raw_response=raw, http_status=exc.code,
                provider_request_id=exc.headers.get("x-request-id"),
                retry_after_seconds=_retry_after(exc.headers.get("Retry-After")),
                latency_seconds=time.monotonic() - started,
                error_metadata=_safe_error_metadata(raw),
            )
        except TimeoutError as exc:
            return NormalizedResponse(
                status="timeout", raw_response="", latency_seconds=time.monotonic() - started,
                error_metadata={"type": type(exc).__name__, "message": str(exc)},
            )
        except (urllib.error.URLError, ConnectionError) as exc:
            return NormalizedResponse(
                status="connection_error", raw_response="",
                latency_seconds=time.monotonic() - started,
                error_metadata={"type": type(exc).__name__, "message": str(exc)},
            )


def normalize_openai_response(
    *, raw: str, http_status: int, headers: dict[str, str], latency_seconds: float,
) -> NormalizedResponse:
    normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    try:
        payload = json.loads(raw)
        choice = payload["choices"][0]
        message = choice["message"]
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not a string")
        reasoning_fields = [
            field_name for field_name in ("reasoning", "reasoning_content")
            if _has_reasoning_signal(message.get(field_name))
        ]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        return NormalizedResponse(
            status="malformed_api_response", raw_response=raw, http_status=http_status,
            provider_request_id=normalized_headers.get("x-request-id"), latency_seconds=latency_seconds,
            error_metadata={"type": type(exc).__name__, "message": str(exc)},
        )
    return NormalizedResponse(
        status="success", raw_response=content,
        provider_response_id=str(payload.get("id")) if payload.get("id") is not None else None,
        provider_request_id=normalized_headers.get("x-request-id"), http_status=http_status,
        finish_reason=str(choice.get("finish_reason")) if choice.get("finish_reason") else None,
        latency_seconds=latency_seconds,
        usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
        error_metadata={
            "reasoning_signal_present": bool(reasoning_fields),
            "reasoning_signal_fields": reasoning_fields,
        },
    )


def _has_reasoning_signal(value: Any) -> bool:
    """Detect provider reasoning fields without retaining their hidden contents."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def _retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def normalize_reasoning_content(
    content: str, *, provider: str, model_id: str, reasoning_mode: str,
) -> tuple[str, dict[str, Any]]:
    """Withhold a complete Nebius Qwen reasoning envelope from downstream logs/parsers.

    The final model answer after ``</think>`` is returned byte-for-byte apart
    from surrounding whitespace. Incomplete envelopes are replaced with a
    non-content marker; finish reason still classifies truncation upstream.
    """
    if not (
        provider == "nebius"
        and model_id.startswith("Qwen/Qwen3-")
        and reasoning_mode == "on"
        and content.lstrip().startswith("<think>")
    ):
        return content, {}
    marker = "</think>"
    marker_index = content.find(marker)
    if marker_index < 0:
        return "[WITHHELD_INCOMPLETE_REASONING_TRACE]", {
            "reasoning_trace_present": True,
            "reasoning_trace_withheld": True,
            "reasoning_trace_incomplete": True,
            "reasoning_trace_character_count": len(content),
        }
    trace = content[:marker_index + len(marker)]
    answer = content[marker_index + len(marker):].strip()
    return answer, {
        "reasoning_trace_present": True,
        "reasoning_trace_withheld": True,
        "reasoning_trace_character_count": len(trace),
    }


def _safe_error_metadata(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"message": raw[:500]}
    error = payload.get("error", payload) if isinstance(payload, dict) else payload
    return {"provider_error": error}
