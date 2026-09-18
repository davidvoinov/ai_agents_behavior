"""Native OpenAI Chat Completions Batch adapter for Promise Switch."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import certifi

from ..credentials import require_api_key
from ..schemas import NormalizedResponse
from .batch_transport import (
    BatchRequestMapping, BatchSubmissionError, ProviderBatchStatus,
)
from .transport import REQUEST_USER_AGENT, normalize_openai_response


class OpenAIBatchProvider:
    name = "openai"

    def __init__(
        self, *, base_url: str = "https://api.openai.com/v1/",
        api_key_env: str = "OPENAI_API_KEY", timeout_seconds: float = 90.0,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.urlopen = urlopen
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._statuses: dict[str, Mapping[str, Any]] = {}

    def serialize(self, mapping: BatchRequestMapping) -> dict[str, Any]:
        body = mapping.request.body
        messages = body.get("messages")
        if (
            mapping.request.model_id != "gpt-5.6-luna"
            or not isinstance(messages, list) or len(messages) != 2
            or [message.get("role") for message in messages] != ["system", "user"]
            or body.get("temperature") != 1.0
            or not isinstance(body.get("max_completion_tokens"), int)
            or "max_tokens" in body
            or body.get("reasoning_effort") not in {"none", "medium"}
        ):
            raise ValueError("invalid GPT-5.6 Luna Promise Switch Batch request")
        _reject_extras(body)
        return {
            "custom_id": mapping.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    def submit(self, input_path: Path, *, metadata: Mapping[str, str]) -> str:
        api_key = self._api_key()
        try:
            file_id = self._upload(input_path, api_key)
        except urllib.error.HTTPError as exc:
            raise BatchSubmissionError(
                f"OpenAI Batch create was rejected with HTTP {exc.code}",
                known_not_submitted=True,
            ) from exc
        except Exception as exc:
            raise BatchSubmissionError(
                f"OpenAI input upload failed: {type(exc).__name__}",
                known_not_submitted=True,
            ) from exc
        body = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": dict(metadata),
        }
        try:
            payload = self._json("POST", "/batches", api_key, body)
        except Exception as exc:
            raise BatchSubmissionError(
                f"OpenAI Batch create outcome is ambiguous: {type(exc).__name__}",
                known_not_submitted=False,
            ) from exc
        job_id = payload.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise BatchSubmissionError(
                "OpenAI Batch create returned no job ID", known_not_submitted=False
            )
        return job_id

    def poll(self, provider_job_id: str) -> ProviderBatchStatus:
        raw = self._json("GET", f"/batches/{provider_job_id}", self._api_key())
        self._statuses[provider_job_id] = raw
        state = str(raw.get("status", "unknown"))
        terminal = state in {"completed", "failed", "expired", "cancelled"}
        return ProviderBatchStatus(state, terminal, state == "completed", raw)

    def retrieve(
        self, provider_job_id: str, status: ProviderBatchStatus
    ) -> Sequence[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for field in ("output_file_id", "error_file_id"):
            file_id = status.raw.get(field)
            if not file_id:
                continue
            raw = self._bytes("GET", f"/files/{file_id}/content", self._api_key())
            for line in raw.decode("utf-8").splitlines():
                if line.strip():
                    rows.append(_strip_openai_hidden_reasoning(json.loads(line)))
        return rows

    def normalize(self, result: Mapping[str, Any]) -> NormalizedResponse:
        error = result.get("error")
        if error:
            text = json.dumps(error, ensure_ascii=False)
            lowered = text.lower()
            return NormalizedResponse(
                status="timeout" if "timeout" in lowered else "error",
                raw_response="", http_status=_error_status(error),
                error_metadata={"provider_error": error},
            )
        response = result.get("response")
        if not isinstance(response, Mapping) or not isinstance(response.get("body"), Mapping):
            return NormalizedResponse(
                status="malformed_api_response", raw_response="",
                error_metadata={"type": "missing_batch_response"},
            )
        headers = response.get("headers") if isinstance(response.get("headers"), Mapping) else {}
        normalized = normalize_openai_response(
            raw=json.dumps(response["body"], ensure_ascii=False),
            http_status=int(response.get("status_code", 200)),
            headers={str(key): str(value) for key, value in headers.items()},
            latency_seconds=0.0,
        )
        request_id = response.get("request_id")
        telemetry = result.get("_promise_switch_reasoning", {})
        return NormalizedResponse(
            **{
                **normalized.to_dict(),
                "returned_model_id": (
                    str(response["body"].get("model"))
                    if response["body"].get("model") else None
                ),
                "provider_request_id": (
                    str(request_id) if request_id is not None
                    else normalized.provider_request_id
                ),
                "error_metadata": {
                    **normalized.error_metadata,
                    "reasoning_signal_present": bool(
                        telemetry.get("reasoning_present")
                        or telemetry.get("reasoning_token_count")
                    ),
                    "reasoning_token_count": telemetry.get("reasoning_token_count"),
                },
            }
        )

    def _api_key(self) -> str:
        return require_api_key(self.api_key_env)

    def _upload(self, path: Path, api_key: str) -> str:
        boundary = f"promise-switch-{uuid.uuid4().hex}"
        data = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n"
        ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        raw = self._request(
            "POST", "/files", api_key, data,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        payload = json.loads(raw)
        if not isinstance(payload, Mapping) or not payload.get("id"):
            raise ValueError("upload returned no file ID")
        return str(payload["id"])

    def _json(
        self, method: str, path: str, api_key: str, body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        raw = self._request(
            method, path, api_key,
            json.dumps(body).encode() if body is not None else None,
            content_type="application/json",
        )
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("provider returned a non-object JSON envelope")
        return payload

    def _bytes(self, method: str, path: str, api_key: str) -> bytes:
        return self._request(method, path, api_key, None, content_type="application/json")

    def _request(
        self, method: str, path: str, api_key: str, data: bytes | None,
        *, content_type: str,
    ) -> bytes:
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": content_type,
                "User-Agent": REQUEST_USER_AGENT,
            },
        )
        with self.urlopen(
            request, timeout=self.timeout_seconds, context=self.ssl_context
        ) as response:
            return response.read()


def _reject_extras(body: Mapping[str, Any]) -> None:
    if any(key in body for key in ("tools", "tool_choice", "response_format")):
        raise ValueError("tools and response formatting are forbidden by the frozen protocol")


def _error_status(error: Any) -> int | None:
    if isinstance(error, Mapping):
        for key in ("status_code", "http_status", "code"):
            try:
                return int(error[key])
            except (KeyError, TypeError, ValueError):
                pass
    return None


def _strip_openai_hidden_reasoning(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    present = False
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            for field in ("reasoning", "reasoning_content"):
                if field in message:
                    present = present or bool(message.get(field))
                    message.pop(field, None)
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    details = usage.get("completion_tokens_details", {}) if isinstance(usage, Mapping) else {}
    count = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
    row["_promise_switch_reasoning"] = {
        "reasoning_present": present or bool(count),
        "reasoning_token_count": count,
    }
    return row
